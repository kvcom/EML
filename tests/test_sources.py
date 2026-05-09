from __future__ import annotations

from datetime import date

from bs4 import BeautifulSoup
import httpx

from euromillions.sources.euro_millions_com import EuroMillionsComSource


def test_euro_millions_com_parses_ordinal_draw_date() -> None:
    html = """
    <html>
      <body>
        <h2>Friday's Result</h2>
        <p>- 8 th May 2026</p>
        <ul class="balls">
          <li class="resultBall ball">2</li>
          <li class="resultBall ball">17</li>
          <li class="resultBall ball">19</li>
          <li class="resultBall ball">34</li>
          <li class="resultBall ball">37</li>
          <li class="resultBall lucky-star">8</li>
          <li class="resultBall lucky-star">11</li>
        </ul>
      </body>
    </html>
    """
    soup = BeautifulSoup(html, "html.parser")

    assert EuroMillionsComSource._parse_draw_date(soup) == date(2026, 5, 8)


def test_euro_millions_com_fetches_first_result_ball_block(monkeypatch) -> None:
    html = """
    <html>
      <body>
        <div class="numbers">
          <span class="ball">3</span><span class="ball">4</span>
        </div>
        <h2>Friday's Result</h2>
        <p>- 8 th May 2026</p>
        <ul class="balls">
          <li class="resultBall ball">2</li>
          <li class="resultBall ball">17</li>
          <li class="resultBall ball">19</li>
          <li class="resultBall ball">34</li>
          <li class="resultBall ball">37</li>
          <li class="resultBall ball">8</li>
          <li class="resultBall ball">11</li>
        </ul>
      </body>
    </html>
    """

    def fake_get(url: str, timeout: float) -> httpx.Response:
        _ = timeout
        return httpx.Response(200, text=html, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)

    rows = EuroMillionsComSource().fetch_latest()

    assert len(rows) == 1
    assert rows[0].draw_date == date(2026, 5, 8)
    assert rows[0].mains == (2, 17, 19, 34, 37)
    assert rows[0].stars == (8, 11)
