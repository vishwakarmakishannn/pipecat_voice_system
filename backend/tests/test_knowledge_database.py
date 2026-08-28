import asyncio
import uuid

import pytest
from sqlalchemy import delete, func
from sqlalchemy.future import select

from core.database import AsyncSessionLocal
from core.models import (
    KnowledgeRelease,
    KnowledgeReleaseUnit,
    KnowledgeSource,
    KnowledgeUnit,
)
from services.knowledge.releases import (
    create_release,
    publish_release,
    rollback_release,
    validate_release,
)
from services.knowledge.retrieval import retrieve_knowledge
from services.knowledge.units import UnitInput, approve_unit, upsert_draft_unit


@pytest.mark.database
def test_release_publish_retrieve_and_rollback_round_trip():
    async def exercise():
        prefix = f"integration-{uuid.uuid4().hex[:10]}"
        async with AsyncSessionLocal() as db:
            active = await db.execute(
                select(KnowledgeRelease.id).where(KnowledgeRelease.status == "published")
            )
            if active.scalar_one_or_none() is not None:
                pytest.skip("release lifecycle integration test requires a database with no active release")
            source = KnowledgeSource(
                name=prefix,
                canonical_uri=f"https://www.mswipe.com/{prefix}",
                source_type="website",
                authority=5,
            )
            db.add(source)
            await db.flush()
            first = await upsert_draft_unit(
                db,
                UnitInput(
                    stable_key=f"{prefix}.activation",
                    unit_type="procedure",
                    title="Activate integration terminal",
                    question="How do I activate the integration terminal?",
                    answer="Switch on the integration terminal and follow activation.",
                    retrieval_text="activate integration terminal Mswipe POS",
                    source_uri=source.canonical_uri,
                    source_label=prefix,
                    authority=5,
                ),
            )
            await db.commit()
            await approve_unit(db, first.id)
            release_one = await create_release(
                db, version=f"{prefix}.1", unit_ids=[first.id]
            )
            validation = await validate_release(db, release_one.id)
            assert validation.valid is True
            await publish_release(db, release_one.id)

            response = await retrieve_knowledge(
                "How do I activate the integration terminal?", db=db
            )
            assert response.status == "ok"
            assert response.release_version == f"{prefix}.1"
            assert response.hits[0].stable_key == f"{prefix}.activation"

            revised = await upsert_draft_unit(
                db,
                UnitInput(
                    stable_key=f"{prefix}.activation",
                    unit_type="procedure",
                    title="Activate integration terminal",
                    question="How do I activate the integration terminal?",
                    answer="Use the revised integration activation procedure.",
                    retrieval_text="activate integration terminal Mswipe POS revised",
                    source_uri=source.canonical_uri,
                    source_label=prefix,
                    authority=5,
                ),
            )
            await db.commit()
            assert revised.id != first.id
            assert revised.stable_key == f"{prefix}.activation.v2"
            assert first.answer == "Switch on the integration terminal and follow activation."
            await approve_unit(db, revised.id)
            first.status = "retired"
            await db.commit()
            historical = await retrieve_knowledge(
                "How do I activate the integration terminal?", db=db
            )
            assert historical.status == "ok"
            assert historical.hits[0].stable_key == f"{prefix}.activation"

            unchanged_retired = await upsert_draft_unit(
                db,
                UnitInput(
                    stable_key=f"{prefix}.activation",
                    unit_type="procedure",
                    title="Activate integration terminal",
                    question="How do I activate the integration terminal?",
                    answer="Switch on the integration terminal and follow activation.",
                    retrieval_text="activate integration terminal Mswipe POS",
                    source_uri=source.canonical_uri,
                    source_label=prefix,
                    authority=5,
                ),
            )
            assert unchanged_retired.id == first.id
            assert unchanged_retired.status == "retired"

            second = await upsert_draft_unit(
                db,
                UnitInput(
                    stable_key=f"{prefix}.contact",
                    unit_type="contact_information",
                    title="Integration support contact",
                    answer="Use the approved integration support channel.",
                    retrieval_text="integration support contact channel",
                    source_uri=source.canonical_uri,
                    source_label=prefix,
                    authority=5,
                ),
            )
            await db.commit()
            await approve_unit(db, second.id)
            release_two = await create_release(
                db, version=f"{prefix}.2", unit_ids=[revised.id, second.id]
            )
            await publish_release(db, release_two.id)
            await rollback_release(db, release_one.id)
            refreshed = await db.execute(
                select(KnowledgeRelease).where(KnowledgeRelease.id == release_one.id)
            )
            assert refreshed.scalar_one().status == "published"

            # Clean only records created by this test, in dependency order.
            release_ids = [release_one.id, release_two.id]
            await db.execute(
                delete(KnowledgeReleaseUnit).where(
                    KnowledgeReleaseUnit.release_id.in_(release_ids)
                )
            )
            await db.execute(
                delete(KnowledgeRelease).where(KnowledgeRelease.id.in_(release_ids))
            )
            await db.execute(
                delete(KnowledgeUnit).where(
                    KnowledgeUnit.id.in_([first.id, revised.id, second.id])
                )
            )
            await db.execute(delete(KnowledgeSource).where(KnowledgeSource.id == source.id))
            await db.commit()

    asyncio.run(exercise())
