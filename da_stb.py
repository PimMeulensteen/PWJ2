from typing import List
from random import sample


def da_stb(schools: List[int], students: List[List[int]]) -> List[int]:
    used = [0] * len(schools)
    placement = []
    for student in sample(students, len(students)):
        placed = False
        for rank, school_at_rank in enumerate(student):
            if used[school_at_rank] < schools[school_at_rank]:
                placement.append(rank + 1)
                used[school_at_rank] += 1
                placed = True
                break
        if not placed:
            placement.append(13)
    return placement
