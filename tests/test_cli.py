"""From a freqtrade hyperopt to a live verdict without writing Python."""

from __future__ import annotations

from pathlib import Path

import pytest

from trialgate.__main__ import main

FIXTURE = Path(__file__).parent / "fixtures" / "freqtrade"


def test_hyperopt_to_verdict_to_live_bands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = str(tmp_path / "registry.jsonl")

    assert main(["import-freqtrade", str(FIXTURE), "--registry", registry]) == 0
    imported = capsys.readouterr().out
    assert "registered 8 new trials (1-8)" in imported
    assert "freqtrade's own pick (lowest loss) is trial" in imported
    pick = int(imported.split("is trial ")[1].split(",")[0])

    verdict = main(
        [
            "evaluate",
            registry,
            "--trial",
            str(pick),
            "--periods-per-year",
            "365",
            "--effective-trials",
            "8",
        ]
    )
    report = capsys.readouterr().out
    assert verdict == 1  # eight epochs of a sample strategy over six months are not evidence
    assert "data floor (gate 2)" in report
    assert "cost stress (gate 9)" in report
    assert "at 2x the costs" in report
    assert report.rstrip().endswith("NOT QUALIFIED")

    judge = ["evaluate", registry, "--trial", str(pick), "--periods-per-year", "365"]
    main([*judge, "--effective-trials", "8", "--cost-multiplier", "3"])
    assert "at 3x the costs" in capsys.readouterr().out
    main([*judge, "--effective-trials", "8", "--no-cost-stress"])
    assert "gate 9" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main([*judge, "--effective-trials", "8", "--no-cost-stress", "--cost-multiplier", "3"])
    capsys.readouterr()

    bands = str(tmp_path / "live_bands.json")
    lock_args = ["live-lock", registry, "--trial", str(pick), "--horizon", "30", "--out", bands]
    assert main(lock_args) == 0
    assert "LOCKED  gate 6 for trial" in capsys.readouterr().out

    calm = tmp_path / "live.csv"
    calm.write_text("date,return\n2026-11-01,0.0\n2026-11-02,0.0001\n", encoding="utf-8")
    assert main(["live-check", bands, str(calm), "--registry", registry]) == 0
    assert capsys.readouterr().out.startswith("PENDING  live deviation (gate 6): 2 of 30")

    crash = tmp_path / "crash.json"
    crash.write_text("[-0.2, -0.2, -0.2]", encoding="utf-8")
    assert main(["live-check", bands, str(crash)]) == 1
    assert capsys.readouterr().err.startswith("STOP  live deviation (gate 6)")
