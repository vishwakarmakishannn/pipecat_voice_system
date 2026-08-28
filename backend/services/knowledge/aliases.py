from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from core.models import KnowledgeAlias


DEFAULT_MS_WIPE_ALIASES = (
    ("Mswipe", "M swipe", "brand", None, 100),
    ("Mswipe", "em swipe", "stt", None, 90),
    ("WisePOS", "wise pos", "product", "WisePOS", 100),
    ("WisePOS", "vice pos", "stt", "WisePOS", 80),
    ("YPOS", "why pos", "stt", "YPOS", 80),
    ("Boombox", "boom box", "product", "Boombox", 100),
    ("Boombox", "sound box", "product", "Boombox", 70),
    ("Neo 2", "neo two", "stt", "Neo 2", 90),
    ("Neo 2", "neo to", "stt", "Neo 2", 70),
    ("NFC", "tap payment", "concept", None, 70),
    ("NFC", "contactless", "concept", None, 70),
)


async def seed_default_aliases(db: AsyncSession) -> int:
    rows = [
        {
            "canonical": canonical,
            "alias": alias,
            "alias_type": alias_type,
            "product": product,
            "language": "en",
            "priority": priority,
            "active": True,
        }
        for canonical, alias, alias_type, product, priority in DEFAULT_MS_WIPE_ALIASES
    ]
    statement = insert(KnowledgeAlias).values(rows)
    statement = statement.on_conflict_do_nothing(
        index_elements=["canonical", "alias", "language"]
    )
    result = await db.execute(statement)
    await db.commit()
    return result.rowcount or 0
