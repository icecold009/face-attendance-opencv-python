"""Unit tests for AttendanceSystem."""

import os
import pytest
import pandas as pd

from attendance import AttendanceSystem


@pytest.fixture
def system(tmp_path):
    """Return a fresh AttendanceSystem backed by a temporary directory."""
    return AttendanceSystem(str(tmp_path))


# ---------------------------------------------------------------------------
# get_attendance_file
# ---------------------------------------------------------------------------

def test_get_attendance_file_creates_csv(system, tmp_path):
    filepath = system.get_attendance_file()
    assert os.path.exists(filepath)
    assert filepath.endswith('.csv')


def test_get_attendance_file_has_correct_columns(system):
    filepath = system.get_attendance_file()
    df = pd.read_csv(filepath)
    assert list(df.columns) == ['Name', 'Time', 'Status']


def test_get_attendance_file_idempotent(system):
    """Calling twice should return the same path and not overwrite data."""
    path1 = system.get_attendance_file()
    system.mark_attendance('Alice')
    path2 = system.get_attendance_file()
    assert path1 == path2
    df = pd.read_csv(path1)
    assert len(df) == 1


# ---------------------------------------------------------------------------
# mark_attendance
# ---------------------------------------------------------------------------

def test_mark_attendance_returns_true_first_time(system):
    assert system.mark_attendance('Alice') is True


def test_mark_attendance_records_name(system):
    system.mark_attendance('Bob')
    df = pd.read_csv(system.get_attendance_file())
    assert 'Bob' in df['Name'].values


def test_mark_attendance_records_status(system):
    system.mark_attendance('Carol', status='Late')
    df = pd.read_csv(system.get_attendance_file())
    assert df.loc[df['Name'] == 'Carol', 'Status'].iloc[0] == 'Late'


def test_mark_attendance_duplicate_returns_false(system):
    system.mark_attendance('Dave')
    assert system.mark_attendance('Dave') is False


def test_mark_attendance_duplicate_not_written_twice(system):
    system.mark_attendance('Eve')
    system.mark_attendance('Eve')
    df = pd.read_csv(system.get_attendance_file())
    assert len(df[df['Name'] == 'Eve']) == 1


def test_mark_attendance_unknown_returns_false(system):
    assert system.mark_attendance('Unknown') is False


def test_mark_attendance_unknown_not_written(system):
    system.mark_attendance('Unknown')
    df = pd.read_csv(system.get_attendance_file())
    assert 'Unknown' not in df['Name'].values


def test_mark_attendance_multiple_people(system):
    system.mark_attendance('Alice')
    system.mark_attendance('Bob')
    df = pd.read_csv(system.get_attendance_file())
    assert set(df['Name'].values) == {'Alice', 'Bob'}


# ---------------------------------------------------------------------------
# reset_daily_marked
# ---------------------------------------------------------------------------

def test_reset_daily_marked_clears_set(system):
    system.mark_attendance('Alice')
    assert 'Alice' in system.marked_today
    system.reset_daily_marked()
    assert len(system.marked_today) == 0


def test_mark_after_reset_succeeds(system):
    system.mark_attendance('Alice')
    system.reset_daily_marked()
    assert system.mark_attendance('Alice') is True


# ---------------------------------------------------------------------------
# get_attendance_summary
# ---------------------------------------------------------------------------

def test_get_attendance_summary_returns_dataframe(system):
    system.mark_attendance('Alice')
    summary = system.get_attendance_summary()
    assert isinstance(summary, pd.DataFrame)
    assert 'Alice' in summary['Name'].values


def test_get_attendance_summary_empty_file(system):
    summary = system.get_attendance_summary()
    assert isinstance(summary, pd.DataFrame)
    assert summary.empty
