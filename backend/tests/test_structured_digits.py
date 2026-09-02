from services.structured_digits import extract_digit_sequences, unique_sequence_for_length


def test_plain_spoken_digits_preserve_exact_order_and_cardinality():
    sequence = unique_sequence_for_length(
        "one two three four five six seven eight nine zero",
        10,
    )

    assert sequence is not None
    assert sequence.value == "1234567890"


def test_repetition_operator_expands_only_the_following_digit():
    sequence = unique_sequence_for_length(
        "nine double zero four eight zero one eight zero six",
        10,
    )

    assert sequence is not None
    assert sequence.value == "9004801806"


def test_separate_numeric_fields_keep_source_order_and_spans():
    sequences = extract_digit_sequences(
        "Customer C one two three four five six and mobile nine eight seven six "
        "five four three two one zero"
    )

    assert [sequence.value for sequence in sequences] == [
        "123456",
        "9876543210",
    ]
    assert sequences[0].end <= sequences[1].start
