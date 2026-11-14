import pytest

from marketdetective.news import FinnhubClient, NewsArchive, download_event_news, event_news_range

# 2024-04-19 13:00 UTC e 2024-04-19 22:00 UTC (dopo la chiusura)
RAW = [
    {"id": 1, "datetime": 1713531600, "headline": "Chip stocks slide", "summary": "Semis fall.", "source": "Reuters", "url": "u1"},
    {"id": 2, "datetime": 1713564000, "headline": "After-hours note", "summary": "", "source": "CNBC", "url": "u2"},
    {"id": 3, "datetime": 1713531600, "headline": "", "summary": "no headline", "source": "X", "url": "u3"},
]


class FakeResponse:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class FakeSession:
    def __init__(self, data=RAW, status=200):
        self.data, self.status, self.calls = data, status, []

    def get(self, url, params=None, timeout=None):
        self.calls.append(params)
        return FakeResponse(self.data, self.status)


def test_parses_items_and_skips_empty_headlines():
    items = FinnhubClient("k", session=FakeSession()).company_news("nvda", "2024-04-16", "2024-04-21")
    assert [i.id for i in items] == ["fh-1", "fh-2"]
    assert items[0].ticker == "NVDA"
    assert items[0].published.startswith("2024-04-19T13:00")


def test_sends_expected_params():
    s = FakeSession()
    FinnhubClient("k", session=s).company_news("NVDA", "2024-04-16", "2024-04-21")
    assert s.calls[0] == {"symbol": "NVDA", "from": "2024-04-16", "to": "2024-04-21", "token": "k"}


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FINNHUB_API_KEY"):
        FinnhubClient(session=FakeSession())


def test_bad_key_message():
    with pytest.raises(RuntimeError, match="401"):
        FinnhubClient("k", session=FakeSession(status=401)).company_news("NVDA", "a", "b")


def test_downloads_one_day_at_a_time(tmp_path):
    s = FakeSession()
    rep = download_event_news(FinnhubClient("k", session=s), NewsArchive(tmp_path), "NVDA", "2024-04-19", pause=0)
    assert [(c["from"], c["to"]) for c in s.calls] == [(d, d) for d in
        ["2024-04-16", "2024-04-17", "2024-04-18", "2024-04-19", "2024-04-20", "2024-04-21"]]
    assert rep.downloaded == 12 and rep.added == 2      # la stessa notizia ogni giorno: deduplicata
    assert rep.saturated_days == []


def test_flags_days_at_finnhub_cap(tmp_path):
    many = [{"id": i, "datetime": 1713531600 + i, "headline": f"h{i}", "summary": "", "source": "s", "url": ""}
            for i in range(250)]
    rep = download_event_news(FinnhubClient("k", session=FakeSession(data=many)), NewsArchive(tmp_path),
                              "NVDA", "2024-04-19", days_before=0, days_after=0, pause=0)
    assert rep.saturated_days == ["2024-04-19"]


def test_archive_deduplicates(tmp_path):
    client = FinnhubClient("k", session=FakeSession())
    archive = NewsArchive(tmp_path)
    assert download_event_news(client, archive, "NVDA", "2024-04-19", pause=0).added == 2
    assert download_event_news(client, archive, "NVDA", "2024-04-19", pause=0).added == 0
    assert len(archive.load("NVDA")) == 2


def test_event_range():
    assert event_news_range("2024-04-19", 3, 2) == ("2024-04-16", "2024-04-21")
