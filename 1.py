import pyiri
tec = pyiri.get_tec(2024, 3, 21, 12, 40.0, 116.0, 300.0)
print(f"TEC: {tec:.2f} TECU")