import json
from pathlib import Path


class UnitHandler:
    """
    matplotlib.pyplot 單位
    """
    def __init__(self):
        # 構建相對路徑
        self.__path = Path(__file__).parent / 'units.json'
        self.data = self.__load_jsonfile()

        # bug in Pycharm
        self.shape = ''

    def __call__(self, key):
        try:
            value = self.__dict__[key]
            return r'${}$'.format(value.replace(' ', r'\ '))
        except KeyError:
            print(f"Attribute '{key}' not found.")
            print(f"Please use unit.set_unit(key, value) to set new unit.")
            return None

    def __getattr__(self, item):
        try:
            value = super().__getattribute__(item)
            return r'${}$'.format(value.replace(' ', r'\space'))

        except AttributeError:
            print(f"Attribute '{item}' not found.")
            print(f"Please use unit.set_unit(key, value) to set new unit.")
            return None

    def set_unit(self, key, value):
        self.__dict__[key] = value
        self.data[key] = value
        self.__update_jsonfile()
        print(f"Attribute '{key}' added and updated in JSON file.")

    def __load_jsonfile(self):
        """ 讀取 JSON 檔中數據并將其變成屬性 """
        try:
            with open(self.__path, 'r', encoding='utf-8') as f:
                _data = json.load(f)
                self.__dict__.update(_data)
                return _data

        except FileNotFoundError:
            print(f"JSON file '{self.__path}' not found.")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in '{self.__path}'.")

    def __update_jsonfile(self):
        """ 更新JSON檔 """
        with open(self.__path, 'r', encoding='utf-8') as f:
            old_data = json.load(f)

        old_data.update(self.data)

        with open(self.__path, 'w', encoding='utf-8') as f:
            json.dump(old_data, f, indent=4)


unit = UnitHandler()
