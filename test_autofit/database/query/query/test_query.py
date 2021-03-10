def test_is_complete(
        gaussian_1,
        aggregator
):
    assert aggregator.query(
        aggregator.is_complete
    ) == [gaussian_1]


def test_is_not_complete(
        gaussian_2,
        aggregator
):
    assert aggregator.query(
        ~aggregator.is_complete
    ) == [gaussian_2]


def test_call(
        gaussian_1,
        aggregator
):
    assert aggregator(
        aggregator.is_complete
    ) == [gaussian_1]


def test_query_dataset(
        gaussian_1,
        gaussian_2,
        aggregator
):
    assert aggregator.query(
        aggregator.dataset_name == "dataset 1"
    ) == [gaussian_1]
    assert aggregator.query(
        aggregator.dataset_name == "dataset 2"
    ) == [gaussian_2]
    assert aggregator.query(
        aggregator.dataset_name.contains(
            "dataset"
        )
    ) == [gaussian_1, gaussian_2]


def test_combine(
        aggregator,
        gaussian_1
):
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.centre == 1)
    ) == [gaussian_1]
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 2") & (aggregator.centre == 1)
    ) == []
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.centre == 2)
    ) == []


def test_combine_attributes(
        aggregator,
        gaussian_1,
        gaussian_2
):
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.phase_name == "phase")
    ) == [gaussian_1]
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 2") & (aggregator.phase_name == "phase")
    ) == [gaussian_2]
    assert aggregator.query(
        (aggregator.dataset_name == "dataset 1") & (aggregator.phase_name == "face")
    ) == []
