#!/usr/bin/env python3 

# @relation 'generators.AgrawalGenerator -i 42'
# 
# @attribute salary numeric
# @attribute commission numeric
# @attribute age numeric
# @attribute elevel {level0,level1,level2,level3,level4}
# @attribute car {car1,car2,car3,car4,car5,car6,car7,car8,car9,car10,car11,car12,car13,car14,car15,car16,car17,car18,car19,car20}
# @attribute zipcode {zipcode1,zipcode2,zipcode3,zipcode4,zipcode5,zipcode6,zipcode7,zipcode8,zipcode9}
# @attribute hvalue numeric
# @attribute hyears numeric
# @attribute loan numeric
# @attribute class {groupA,groupB}

import os
import sys

filename = sys.argv[1]

with open(filename, "r") as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]

with open(filename, "w") as out:
    for line in lines:
        new_line = []
        attributes = [x.strip() for x in line.split(',')]

        salary = int(float(attributes[0]) / 10000) # 20,000 - 150,000
        for i in range(2, 16):
            new_line.append('1' if i == salary else '0')

        commission = int(float(attributes[1]) / 10000)
        for i in range(0, 7):
            new_line.append('1' if i == commission else '0')

        age = int(float(attributes[2]) / 10) # [20, 80]
        for i in range(2, 9):
            new_line.append('1' if age == i else '0')

        level = int(attributes[3][5:]) # [0, 4]
        for i in range(0, 5):
            new_line.append('1' if level == i else '0')

        car = int(attributes[4][3:])
        for i in range(1, 20):
            new_line.append('1' if i == car else '0')

        zipcode = int(attributes[5][7:])
        for i in range(1, 9):
            new_line.append('1' if i == zipcode else '0')

        hvalue = int(float(attributes[6]) / 10000)
        for i in range(0, 15):
            new_line.append('1' if i == hvalue else '0')

        hyears = int(float(attributes[7]) / 10)
        for i in range(1, 3):
            new_line.append('1' if i == hyears else '0')

        loan = int(float(attributes[8]) / 100000)
        for i in range(0, 5):
            new_line.append('1' if i == loan else '0')

        group = attributes[9]
        new_line.append('0' if group == "groupA" else '1')

        new_line.append("\n")
        out.write(','.join(new_line))
        out.flush()
