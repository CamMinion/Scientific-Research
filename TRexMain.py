import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

MIN_VAL = -10
MAX_VAL = 10

class Prey:
    def __init__(self):
        self.x = [0.0] * 200
        self.f = 0.0
        self.f2 = 0.0

import numpy as np

def rastrigin_func(x):
    return 10 * len(x) + sum((xi**2 - 10 * np.cos(2 * np.pi * xi)) for xi in x)

def return_fitness_kl(a, n):
    a.f = rastrigin_func(a.x[:n])
    return a.f

def return_fitness2_kl(a, n):
    a.f2 = rastrigin_func(a.x[:n])
    return a.f2

def return_fitness_tg(a, kl, n):
    a.f = abs(kl.f - rastrigin_func(a.x[:n]))
    return a.f

def return_fitness2_tg(a, kl, n):
    a.f2 = abs(kl.f2 - rastrigin_func(a.x[:n]))
    return a.f2



def random_float():
    return MIN_VAL + (MAX_VAL - MIN_VAL) * random.random()

"""def return_fitness_kl(a, n):
    a.f = sum(a.x[i] ** 2 for i in range(n))
    return a.f

def return_fitness2_kl(a, n):
    a.f2 = sum(a.x[i] ** 2 for i in range(n))
    return a.f2

def return_fitness_tg(a, kl, n):
    a.f = abs(kl.f - sum(a.x[i] ** 2 for i in range(n)))
    return a.f

def return_fitness2_tg(a, kl, n):
    a.f2 = abs(kl.f2 - sum(a.x[i] ** 2 for i in range(n)))
    return a.f2"""

def khoi_tao_vi_tri(a, n):
    for i in range(6):
        for j in range(n):
            a[i].x[j] = random_float()
    a[0].f = return_fitness_kl(a[0], n)
    for i in range(1, 6):
        a[i].f = return_fitness_tg(a[i], a[0], n)

def xuat(a):
    print("\t\t\tFITNESS 1 \t\t\t\tFITNESS 2")
    print(f"\tT- rex : {a[0].f}\t\t\t\t{a[0].f2}")
    for i in range(1, 6):
        print(f"\tPrey {i} : {a[i].f}\t\t\t\t{a[i].f2}")

def chinh_lai_toa_do(a):
    if a > 10:
        return 10 - (a - 10)
    elif a < -10:
        return -10 + (-10 - a)
    return a

"""def cap_nhap_vi_tri_kl(kl, target, n):
    for i in range(n):
        kl.x[i] = kl.x[i] + (target.x[i] - kl.x[i]) * random.random()
        kl.x[i] = chinh_lai_toa_do(kl.x[i])"""

def cap_nhap_vi_tri_kl(kl, target, n):
    for i in range(n):
        kl.x[i] = kl.x[i] + (target.x[i] - kl.x[i]) * random.uniform(0.5, 1.5)  # Thay đổi ngẫu nhiên trong khoảng 0.5 đến 1.5
        kl.x[i] = chinh_lai_toa_do(kl.x[i])       

def dich_chuyen_tuc_thoi(a, n):
    khoang_cach = 1.0
    for i in range(n):
        x_min = a.x[i] - khoang_cach
        x_max = a.x[i] + khoang_cach
        a.x[i] = random.uniform(x_min,x_max)
        a.x[i] = chinh_lai_toa_do(a.x[i])

def cap_nhap_vi_tri_ntg(a, n):
    for i in range(n):
        a.x[i] = a.x[i] - ((MAX_VAL-(-MIN_VAL))/MAX_VAL) * random.uniform(-1,1)
        a.x[i] = chinh_lai_toa_do(a.x[i])

def tim_min_fitness(a):
    return min(range(1, 6), key=lambda i: a[i].f)

def tim_max_fitness(a):
    return max(range(1, 6), key=lambda i: a[i].f)

def tao_con_moi_sau_khi_con_cu_chet(a, b, n):
    x = Prey()
    x.f = b.f
    for _ in range(200):
        c = tim_min_fitness(a)
        for i in range(n):
            x.x[i] = random.uniform(a[c].x[i]*random.random() + a[c].x[i],a[c].x[i] - a[c].x[i]*random.random())
            x.x[i] = chinh_lai_toa_do(x.x[i])
        if return_fitness2_tg(a[c], a[0], n) < a[tim_max_fitness(a)].f:
            return x
    print('Qua 200 lan lap ko cai thien fitness')
    return x

def cap_nhap_vi_tri_tg(a, kl, n):
    for i in range(n):
        a.x[i] = a.x[i] + (a.x[i] - kl.x[i]) * random.uniform(-1, 1)
        a.x[i] = chinh_lai_toa_do(a.x[i])

def chon_target(a):
    target = tim_min_fitness(a)
    print(f"      ==>Con Target la con Prey {target} - Fitness  {a[target].f}")
    return target

def chuyen_doi_tu_f2_sang_f1(a):
    for prey in a:
        prey.f = prey.f2
        prey.f2 = 0

def sao_chep(a, b):
    for i in range(6):
        b[i].x = a[i].x.copy()
        b[i].f = a[i].f
        b[i].f2 = a[i].f2

# Hàm cập nhật dữ liệu cho đồ thị với chỉ 1 đường best fitness
def update_plot(best_fitness_history, ax):
    ax.clear()  # Xóa đồ thị cũ
    ax.plot(best_fitness_history, label='Best Fitness', color='blue')
    ax.legend(loc="upper right")
    ax.set_xlabel('Vòng lặp')
    ax.set_ylabel('Fitness (log scale)')
    ax.set_title('Lịch sử Best Fitness qua các vòng lặp')

    # Sử dụng trục y logarit để dễ quan sát fitness nhỏ
    ax.set_yscale('log')
   

def main():
    random.seed()
    a = [Prey() for _ in range(6)]
    b = [Prey() for _ in range(6)]
    target = 0

    print("Hay chon so chieu:")
    n = int(input())
    khoi_tao_vi_tri(a, n)
    print("Vi tri khoi tao ban dau la:")
    xuat(a)

    print("Hay nhap so vong ban muon thuc hien:")
    number = int(input())
    
    # Chuẩn bị cho đồ thị chỉ với Best Fitness
    best_fitness_history = []  # Lưu lịch sử best fitness
    fig, ax = plt.subplots()
    
    for loop_n in range(number):
        target = chon_target(a)
        sao_chep(a, b)  # Lưu dữ liệu ban đầu
        cap_nhap_vi_tri_kl(a[0], a[target], n)
        nho = a[target].f  # Biến "nho" sẽ lưu fitness cũ để so sánh

        choice = random.random()
        if choice < 0.4:  # TH target bị ẩn
            a[0].x = a[target].x.copy()
            a[target] = tao_con_moi_sau_khi_con_cu_chet(a, a[target], n)
            print("Target bị ẩn")
        else:  # TH target chạy
            choice = random.random()
            if choice < 0.5:
                dich_chuyen_tuc_thoi(a[target], n)
            else:
                cap_nhap_vi_tri_tg(a[target], a[0], n)
        
        # Cập nhật vị trí và tính fitness cho các non-target
        for i in range(1, 6):
            h = random.random()
            if h <= 0.2:
                a[i] = tao_con_moi_sau_khi_con_cu_chet(a, a[i], n)
                print(f"Có non-target {i} bị trúng độc")
            else:
                cap_nhap_vi_tri_ntg(a[i], n)
        
        # Tính Fitness 2 
        a[0].f2 = return_fitness2_kl(a[0], n)
        for i in range(1, 6):
            a[i].f2 = return_fitness2_tg(a[i], a[0], n)

        # Lưu lại giá trị best fitness sau mỗi vòng lặp
        best_fitness = min(prey.f for prey in a)
        best_fitness_history.append(best_fitness)

        # Vẽ lại đồ thị
        update_plot(best_fitness_history, ax)
        plt.pause(0.5)  # Tạm dừng để hiển thị đồ thị
        
        # Kiểm tra cải thiện fitness
        j = min(range(1, 6), key=lambda i: a[i].f2)
        if nho > a[j].f2:
            chuyen_doi_tu_f2_sang_f1(a)
            print(f"----------------------------Xong vòng {loop_n}----------------------------\n\n")
        else:
            print("Không cải thiện, làm lại bước này\n")
            sao_chep(b, a)
            print(f"----------------------------Xong vòng {loop_n}----------------------------\n\n")
        xuat(a)

    plt.show()

if __name__ == "__main__":
    main()
