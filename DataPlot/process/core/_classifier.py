

class Classifier:

    def __new__(cls, df):
        pass

    @classmethod
    def classify(cls, df):
        pass

    @staticmethod
    def map_diurnal(hour):
        if 7 <= hour <= 18:
            return 'Day'
        elif 19 <= hour <= 23:
            return 'Night'
        elif 0 <= hour <= 6:
            return 'Night'

    @staticmethod
    def map_state(row, clean_upp_boud, event_low_boud):
        if row['Extinction'] >= event_low_boud:
            return 'Event'
        elif clean_upp_boud < row['Extinction'] < event_low_boud:
            return 'Transition'
        else:
            return 'Clean'

