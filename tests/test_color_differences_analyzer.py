from typer.testing import CliRunner

from color_differences_analyzer.main import app

runner = CliRunner()


def test_color_differences_analyzer():
    result = runner.invoke(app)
    assert result.exit_code == 0
    assert "Hello World!" in result.stdout
