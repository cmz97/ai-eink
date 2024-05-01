import smbus
import time
import struct

class BQ27441:
    def __init__(self) -> None:
        self.bus=smbus.SMBus(1)
        self.address = 0x55

    def getVoltage(self):
        data = self.bus.read_i2c_block_data(self.address,0x02,2)
        (val,)=struct.unpack('H', bytearray(data)[0:2])
        print(f"Voltage: {val}mV")
