"""Low-latency phrase aggregation for local Kokoro synthesis."""

from collections.abc import AsyncIterator

from pipecat.utils.text.base_text_aggregator import (
    Aggregation,
    AggregationType,
    BaseTextAggregator,
)


_STRONG_BOUNDARIES = frozenset(".!?")
_CLAUSE_BOUNDARIES = frozenset(",;:")


class KokoroTextAggregator(BaseTextAggregator):
    """Release a short first phrase, then larger chunks to keep audio buffered."""

    def __init__(
        self,
        *,
        first_chunk_chars: int,
        first_chunk_min_words: int,
        chunk_chars: int,
        min_chunk_words: int,
    ):
        super().__init__(aggregation_type=AggregationType.SENTENCE)
        self._first_chunk_chars = first_chunk_chars
        self._first_chunk_min_words = first_chunk_min_words
        self._chunk_chars = chunk_chars
        self._min_chunk_words = min_chunk_words
        self._text = ""
        self._emitted_chunk = False

    @property
    def text(self) -> Aggregation:
        return Aggregation(
            text=self._text.strip(),
            type=AggregationType.SENTENCE,
        )

    async def aggregate(self, text: str) -> AsyncIterator[Aggregation]:
        for char in text:
            self._text += char
            if self._should_emit(char):
                aggregation = self._take_buffer()
                if aggregation:
                    yield aggregation

    def _should_emit(self, char: str) -> bool:
        stripped = self._text.strip()
        if not stripped:
            return False

        if char in _STRONG_BOUNDARIES:
            return True

        word_count = len(stripped.split())
        min_words = (
            self._min_chunk_words
            if self._emitted_chunk
            else self._first_chunk_min_words
        )
        if char in _CLAUSE_BOUNDARIES and word_count >= min_words:
            return True

        target_chars = (
            self._chunk_chars if self._emitted_chunk else self._first_chunk_chars
        )
        return (
            char.isspace()
            and len(stripped) >= target_chars
            and word_count >= min_words
        )

    def _take_buffer(self) -> Aggregation | None:
        text = self._text.strip()
        self._text = ""
        if not text:
            return None
        self._emitted_chunk = True
        return Aggregation(text=text, type=AggregationType.SENTENCE)

    async def flush(self) -> Aggregation | None:
        aggregation = self._take_buffer()
        self._emitted_chunk = False
        return aggregation

    async def handle_interruption(self):
        await self.reset()

    async def reset(self):
        self._text = ""
        self._emitted_chunk = False
