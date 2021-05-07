import random


def gen_data(n_schools=63, n_students=8000, n_fac=1.3):
    school_cap = [int((n_students * n_fac) / n_schools)] * n_schools
    student_choice = [[] for _ in range(n_students)]
    n_scls_to_choose = min(12, n_schools)

    for i in range(n_students):
        student_choice[i] = random.sample(range(n_schools), n_scls_to_choose)

    return school_cap, student_choice


print(gen_data())
