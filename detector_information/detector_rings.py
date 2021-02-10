# Stores IC string numbers per ring
# Ring 0 is center most string, counting up from center

def create_ring_dict():
    rings = {}
    rings[0] = [36]
    rings[1] = [26, 27, 35, 37, 45, 46]
    rings[2] = [17, 18, 19, 25, 28, 34, 38, 44, 47, 54, 55, 56]
    rings[3] = [9, 10, 11, 12, 16, 20, 24, 29, 33, 39, 43, 48, 53, 57, 62, 63, 64, 65]
    rings[4] = [2,3,4,5,6,8,13,15,21,23,30,32,40,42,49,52,58,61,66,69,70,71,72,73]
    rings[5] = [1, 7, 14, 22, 31, 41, 50, 51, 59, 60, 67, 68, 74, 75, 76, 77, 78]

    return rings
