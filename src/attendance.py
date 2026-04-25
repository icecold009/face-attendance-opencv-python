from __future__ import annotations

import os
from typing import Optional, Set

import pandas as pd

from datetime import datetime
from utils import get_date, get_timestamp, create_directory


class AttendanceSystem:
    def __init__(self, attendance_path: str = 'data/Attendance') -> None:
        self.attendance_path = attendance_path
        create_directory(attendance_path)
        self.marked_today: Set[str] = set()

    def get_attendance_file(self) -> str:
        """Get or create attendance file for today"""
        date = get_date()
        filename = f"{self.attendance_path}/Attendance_{date}.csv"
        
        if not os.path.exists(filename):
            df = pd.DataFrame(columns=['Name', 'Time', 'Status'])
            df.to_csv(filename, index=False)
        
        return filename
    
    def mark_attendance(self, name: str, status: str = 'Present') -> bool:
        """Mark attendance for a person"""
        if name in self.marked_today or name == "Unknown":
            return False
        
        filename = self.get_attendance_file()
        timestamp = get_timestamp()
        
        try:
            df = pd.read_csv(filename)
            new_record = pd.DataFrame({
                'Name': [name],
                'Time': [timestamp],
                'Status': [status]
            })
            df = pd.concat([df, new_record], ignore_index=True)
            df.to_csv(filename, index=False)
            self.marked_today.add(name)
            return True
        except Exception as e:
            print(f"Error marking attendance: {e}")
            return False
    
    def get_attendance_summary(self) -> Optional[pd.DataFrame]:
        """Get attendance summary for today"""
        filename = self.get_attendance_file()
        
        try:
            df = pd.read_csv(filename)
            return df
        except Exception as e:
            print(f"Error reading attendance: {e}")
            return None
    
    def reset_daily_marked(self) -> None:
        """Reset marked attendance for the day"""
        self.marked_today = set()
    
    def get_person_attendance_history(self, name: str, days: int = 30) -> Optional[pd.DataFrame]:
        """Get attendance history for a person"""
        records = []
        
        # Search through attendance files
        for filename in os.listdir(self.attendance_path):
            if filename.endswith('.csv') and filename.startswith('Attendance_'):
                filepath = os.path.join(self.attendance_path, filename)
                try:
                    df = pd.read_csv(filepath)
                    person_records = df[df['Name'] == name]
                    if not person_records.empty:
                        records.append(person_records)
                except:
                    pass
        
        if records:
            combined = pd.concat(records, ignore_index=True)
            return combined
        return None
