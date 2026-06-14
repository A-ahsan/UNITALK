import serial
import time

s = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

time.sleep(2)

print("Waking GRBL...")
s.write(b"\r\n\r\n")
time.sleep(2)

s.flushInput()

commands = [
    "$H"
]

for cmd in commands:
    print("Sending:", cmd)
    s.write((cmd + '\n').encode())
    time.sleep(1)

    while s.in_waiting:
        print(s.readline().decode(errors='ignore').strip())

s.close()
