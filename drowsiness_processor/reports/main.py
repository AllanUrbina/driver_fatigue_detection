import csv
import os
import json
from datetime import datetime


class DrowsinessReports:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.fields = ['timestamp', 'eye_rub_first_hand_report', 'eye_rub_first_hand_count',
                       'eye_rub_first_hand_durations', '|',
                       'eye_rub_second_hand_report', 'eye_rub_second_hand_count', 'eye_rub_second_hand_durations', '|',
                       'flicker_report', 'flicker_count', '|',
                       'micro_sleep_report', 'micro_sleep_count', 'micro_sleep_durations', '|',
                       'pitch_report', 'pitch_count', 'pitch_durations', '|',
                       'yawn_report', 'yawn_count', 'yawn_durations', '|',
                       'phone_report', 'phone_count', 'phone_durations', 'possible_call']

        if not os.path.exists(self.file_name):
            self.create_csv_file()

    def create_csv_file(self):
        # ensure the destination folder exists (e.g. reports/august/) instead
        # of crashing with FileNotFoundError the first time this runs in a
        # different month or a fresh checkout of the repo
        directory = os.path.dirname(self.file_name)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.file_name, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.fields)
            writer.writeheader()

    def main(self, report_data: dict):
        # NOTE: switched from report_data['eye_rub_first_hand']['eye_rub_report'] style
        # direct indexing to .get() with defaults. Phone detection can produce a report
        # on frames where the face mesh failed (control_process == False upstream), so
        # report_data won't always contain every drowsiness key — direct indexing would
        # raise a KeyError in that case.
        if (report_data.get('eye_rub_first_hand', {}).get('eye_rub_report') or
                report_data.get('eye_rub_second_hand', {}).get('eye_rub_report') or
                report_data.get('flicker_and_micro_sleep', {}).get('flicker_report') or
                report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_report') or
                report_data.get('pitch', {}).get('pitch_report') or
                report_data.get('yawn', {}).get('yawn_report') or
                report_data.get('phone', {}).get('phone_report')):
            row = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'eye_rub_first_hand_report': report_data.get('eye_rub_first_hand', {}).get('eye_rub_report', False),
                'eye_rub_first_hand_count': report_data.get('eye_rub_first_hand', {}).get('eye_rub_count', 0),
                'eye_rub_first_hand_durations': report_data.get('eye_rub_first_hand', {}).get('eye_rub_durations', []),
                '|': '|',
                'eye_rub_second_hand_report': report_data.get('eye_rub_second_hand', {}).get('eye_rub_report', False),
                'eye_rub_second_hand_count': report_data.get('eye_rub_second_hand', {}).get('eye_rub_count', 0),
                'eye_rub_second_hand_durations': report_data.get('eye_rub_second_hand', {}).get('eye_rub_durations', []),
                '|': '|',
                'flicker_report': report_data.get('flicker_and_micro_sleep', {}).get('flicker_report', False),
                'flicker_count': report_data.get('flicker_and_micro_sleep', {}).get('flicker_count', 0),
                '|': '|',
                'micro_sleep_report': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_report', False),
                'micro_sleep_count': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_count', 0),
                'micro_sleep_durations': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_durations', []),
                '|': '|',
                'pitch_report': report_data.get('pitch', {}).get('pitch_report', False),
                'pitch_count': report_data.get('pitch', {}).get('pitch_count', 0),
                'pitch_durations': report_data.get('pitch', {}).get('pitch_durations', []),
                '|': '|',
                'yawn_report': report_data.get('yawn', {}).get('yawn_report', False),
                'yawn_count': report_data.get('yawn', {}).get('yawn_count', 0),
                'yawn_durations': report_data.get('yawn', {}).get('yawn_durations', []),
                '|': '|',
                'phone_report': report_data.get('phone', {}).get('phone_report', False),
                'phone_count': report_data.get('phone', {}).get('phone_count', 0),
                'phone_durations': report_data.get('phone', {}).get('phone_durations', []),
                'possible_call': report_data.get('phone', {}).get('possible_call', False),
            }

            with open(self.file_name, mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fields)
                writer.writerow(row)

    def generate_json_report(self, report_data: dict) -> str:
        report_json = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'eye_rub_first_hand': {
                'report': report_data.get('eye_rub_first_hand', {}).get('eye_rub_report', False),
                'count': report_data.get('eye_rub_first_hand', {}).get('eye_rub_count', 0),
                'durations': report_data.get('eye_rub_first_hand', {}).get('eye_rub_durations', [])
            },
            'eye_rub_second_hand': {
                'report': report_data.get('eye_rub_second_hand', {}).get('eye_rub_report', False),
                'count': report_data.get('eye_rub_second_hand', {}).get('eye_rub_count', 0),
                'durations': report_data.get('eye_rub_second_hand', {}).get('eye_rub_durations', [])
            },
            'flicker': {
                'report': report_data.get('flicker_and_micro_sleep', {}).get('flicker_report', False),
                'count': report_data.get('flicker_and_micro_sleep', {}).get('flicker_count', 0)
            },
            'micro_sleep': {
                'report': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_report', False),
                'count': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_count', 0),
                'durations': report_data.get('flicker_and_micro_sleep', {}).get('micro_sleep_durations', [])
            },
            'pitch': {
                'report': report_data.get('pitch', {}).get('pitch_report', False),
                'count': report_data.get('pitch', {}).get('pitch_count', 0),
                'durations': report_data.get('pitch', {}).get('pitch_durations', [])
            },
            'yawn': {
                'report': report_data.get('yawn', {}).get('yawn_report', False),
                'count': report_data.get('yawn', {}).get('yawn_count', 0),
                'durations': report_data.get('yawn', {}).get('yawn_durations', [])
            },
            'phone': {
                'report': report_data.get('phone', {}).get('phone_report', False),
                'count': report_data.get('phone', {}).get('phone_count', 0),
                'durations': report_data.get('phone', {}).get('phone_durations', []),
                'present': report_data.get('phone', {}).get('phone_present', False),
                'possible_call': report_data.get('phone', {}).get('possible_call', False),
            }
        }
        return json.dumps(report_json)