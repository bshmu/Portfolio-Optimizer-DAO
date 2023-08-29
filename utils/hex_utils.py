hex_dict = {"a": 10, "b": 11, "c": 12, "d": 13, "e": 14, "f": 15}
hex_dict_r = {v: k for k, v in hex_dict.items()}

def dec_to_hex(dec):
    temp_hex = ""
    while (dec/16 > 1/16):
        quotient = int(dec / 16)
        remainder  = int(dec % 16)
        hex_digit = hex_dict_r[remainder] if remainder >= 10 else remainder
        temp_hex += str(hex_digit)
        dec /= 16
    hex = "0x" + temp_hex[::-1]
    return hex

def hex_to_dec(hex):
    temp_hex = hex[2:]
    dec = 0
    for i,c in enumerate(temp_hex):
        digit = len(temp_hex) - i - 1
        digit_mul = hex_dict[c] if c in list(hex_dict.keys()) else int(c)
        dec += 16**digit * digit_mul
    return dec

def decimal_to_q8_23(decimal_value):
    # Multiply with 2^23 and round off to get the fixed point representation
    q8_23_value = int(round(decimal_value * (2**23)))
    # Return value within 32-bit representation (for safety)
    return q8_23_value & 0xFFFFFFFF 

def q8_23_to_decimal(q8_23_value):
    # If the number is negative (based on sign bit)
    if q8_23_value & (1 << 31):
        q8_23_value = q8_23_value - (1 << 32)
    return q8_23_value / (2**23)

def decimal_to_q15_16(decimal_value):
    # Multiply with 2^16 and round off to get the fixed point representation
    q15_16_value = int(round(decimal_value * (2**16)))
    # Return value within 32-bit representation (for safety)
    return q15_16_value & 0xFFFFFFFF 

def q15_16_to_decimal(q15_16_value):
    # If the number is negative (based on sign bit)
    if q15_16_value & (1 << 31):
        q15_16_value = q15_16_value - (1 << 32)
    return q15_16_value / (2**16)

assert (10000 == hex_to_dec(dec_to_hex(10000)))
assert (3.14 == round(q8_23_to_decimal(decimal_to_q8_23(3.14)), 2))
assert (3.14 == round(q15_16_to_decimal(decimal_to_q15_16(3.14)), 2))