import random
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

MIN_VAL = -5
MAX_VAL = 5

MUTATION_RATE = 0.5  # Xác suất đột biến

class Prey:
    def __init__(self):
        self.x = [0.0] * 200
        self.f = 0.0
        self.f2 = 0.0


# HÀM TEST 3
# Hàm Rosenbrock (phức tạp hơn, liên tục, không lồi) 
# Global minimum: 0 tại x = [1,1,...,1]
# [−2, 2]
#def rosenbrock_func(x):
#   return sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2 for i in range(len(x) - 1))


# HÀM TEST 1
# Hàm Sphere (đơn giản, liên tục, lồi) 
# Global minimum: 0 tại x = [0,0,...,0]
# [−5, 5]
#def sphere_func(x):
#    return sum(xi ** 2 for xi in x)


# HÀM TEST 2
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

def khoi_tao_vi_tri(a, n):
    for i in range(11):
        for j in range(n):
            a[i].x[j] = random_float()
    a[0].f = return_fitness_kl(a[0], n)
    for i in range(1, 11):
        a[i].f = return_fitness_tg(a[i], a[0], n)

def xuat(a):
    print("\t\t\tFITNESS 1 \t\t\t\tFITNESS 2")
    print(f"\tT-rex : {a[0].f}\t\t\t\t{a[0].f2}")
    for i in range(1, 11):
        print(f"\tPrey {i} : {a[i].f}\t\t\t\t{a[i].f2}")

def chinh_lai_toa_do(a):
    if a > 10:
        return 10 - (a - 10)
    elif a < -10:
        return -10 + (-10 - a)
    return a

def cap_nhap_vi_tri_kl(kl, target, n):
    for i in range(n):
        kl.x[i] = kl.x[i] + (target.x[i] - kl.x[i]) * random.uniform(0, 1)
        kl.x[i] = chinh_lai_toa_do(kl.x[i])       

def dich_chuyen_tuc_thoi(a, n):
    khoang_cach = 1.5
    for i in range(n):
        x_min = a.x[i] - khoang_cach
        x_max = a.x[i] + khoang_cach
        a.x[i] = random.uniform(x_min, x_max)
        a.x[i] = chinh_lai_toa_do(a.x[i])

def cap_nhap_vi_tri_ntg(a, n):
    for i in range(n):
        a.x[i] = a.x[i] - ((MAX_VAL - (-MIN_VAL)) / MAX_VAL) * random.uniform(0, 1)
        a.x[i] = chinh_lai_toa_do(a.x[i])

def tim_min_fitness(a):
    return min(range(1, 11), key=lambda i: a[i].f)

def tim_max_fitness(a):
    return max(range(1, 11), key=lambda i: a[i].f)


# Tìm hai con tốt nhất
def tim_hai_con_tot_nhat(a):
    # Tạo danh sách chứa chỉ số của các con mồi
    indices = list(range(1, 11))
    
    # Sắp xếp danh sách dựa trên giá trị fitness
    sorted_indices = sorted(indices, key=lambda i: a[i].f)

    # Lấy con tốt nhất và con tốt thứ hai
    best = sorted_indices[0]
    second_best = sorted_indices[1]
    
    return best, second_best


def tao_con_moi_sau_khi_con_cu_chet(a, best, second_best, n):
    x = Prey()
    x.f = a[best].f
    for i in range(n):
        x.x[i] = random.uniform(a[best].x[i], a[second_best].x[i])  # Lai ghép giữa hai con
        x.x[i] = chinh_lai_toa_do(x.x[i])
    mutate(x, a, best)  # Thêm đột biến cho con mới
    return x

def cap_nhap_vi_tri_tg(a, kl, n):
    for i in range(n):
        a.x[i] = a.x[i] + (a.x[i] - kl.x[i]) * random.uniform(0, 1)
        a.x[i] = chinh_lai_toa_do(a.x[i])

def chon_target(a):
    target = tim_min_fitness(a)
    print(f"      ==>Con Target là con Prey {target} - Fitness  {a[target].f:.80f}")
    return target

def chuyen_doi_tu_f2_sang_f1(a):
    for prey in a:
        prey.f = prey.f2
        prey.f2 = 0

def sao_chep(a, b):
    for i in range(11):
        b[i].x = a[i].x.copy()
        b[i].f = a[i].f
        b[i].f2 = a[i].f2


# Đột biến
def mutate(prey, a, best):
    for i in range(len(prey.x)):
        if random.random() < MUTATION_RATE:  # Nếu số ngẫu nhiên nhỏ hơn tỉ lệ đột biến
            # Đột biến bằng cách thêm một giá trị ngẫu nhiên
            prey.x[i] += random.uniform(-1, 1)  # Thay đổi giá trị của tọa độ
            prey.x[i] = chinh_lai_toa_do(prey.x[i])  # Đảm bảo tọa độ trong khoảng hợp lệ
        else:
            prey.x[i] = random.uniform(a[best].x[i] - 0.5, a[best].x[i] + 0.5 )


# Hàm cập nhật dữ liệu cho đồ thị với chỉ 1 đường best fitness
def update_plot(best_fitness_history, ax):
    ax.clear()  # Xóa đồ thị cũ
    ax.plot(best_fitness_history, label='TROA-GA Fitness', color='red')
    ax.legend(loc="upper right")
    ax.set_xlabel('Vòng lặp')
    ax.set_ylabel('Fitness (log scale)')
    ax.set_title('Lịch sử Best Fitness qua các vòng lặp')
    ax.set_ylim(auto=True)
    #ax.set_ylim(1e-30, 1e0)  # Thiết lập khoảng giá trị y từ 10^-30 đến 10^0
    ax.set_yscale('log')  # Sử dụng trục y logarit để dễ quan sát fitness nhỏ



def dat_lai_vi_tri(prey, n):
    # Đặt lại vị trí ngẫu nhiên cho prey
    for i in range(n):
        prey.x[i] = random_float()
    prey.f = return_fitness_kl(prey, n)
    prey.f2 = return_fitness2_kl(prey, n)

def tim_hai_con_xau_nhat(a):
    # Tạo danh sách chứa chỉ số của các con mồi
    indices = list(range(1, 11))
    
    # Sắp xếp danh sách dựa trên giá trị fitness (fitness càng lớn càng xấu)
    sorted_indices = sorted(indices, key=lambda i: a[i].f, reverse=True)

    # Lấy con xấu nhất và con xấu thứ hai
    worst = sorted_indices[0]
    second_worst = sorted_indices[1]
    
    return worst, second_worst

def tim_hai_con_xau_nhat(a):
    # Tạo danh sách chứa chỉ số của các con mồi
    indices = list(range(1, 11))
    
    # Sắp xếp danh sách dựa trên giá trị fitness từ cao đến thấp
    sorted_indices = sorted(indices, key=lambda i: a[i].f, reverse=True)

    # Lấy hai con xấu nhất
    worst1 = sorted_indices[0]
    worst2 = sorted_indices[1]
    
    return worst1, worst2

def sinh_con_moi_tu_hai_con_tot(a, best, second_best, n):
    # Tạo con mồi mới bằng cách lai ghép từ hai con tốt nhất
    new_prey = Prey()
    for i in range(n):
        new_prey.x[i] = random.uniform(a[best].x[i], a[second_best].x[i])  # Lai ghép giữa hai con tốt nhất
        new_prey.x[i] = chinh_lai_toa_do(new_prey.x[i])
    mutate(new_prey, a, best)  # Thực hiện đột biến
    new_prey.f = return_fitness_kl(new_prey, n)  # Tính toán lại fitness
    return new_prey

def main():
    random.seed(42)
    a = [Prey() for _ in range(11)]
    b = [Prey() for _ in range(11)]
    target = 0

    print("Hãy chọn số chiều:")
    n = int(input())
    khoi_tao_vi_tri(a, n)
    print("Vị trí khởi tạo ban đầu là:")
    xuat(a)

    random.seed()  # Bỏ cố định seed    
    print("Hãy nhập số vòng bạn muốn thực hiện:")
    number = int(input())
    
    so_vong_khong_cai_thien = 0  # Đếm số vòng không cải thiện
    gioi_han_khong_cai_thien = 100  # Số vòng không cải thiện tối đa trước khi loại bỏ 2 con xấu nhất

    best_fitness_history = []  # Lưu lịch sử best fitness
    fig, ax = plt.subplots()
    
    for loop_n in range(number):
        target = chon_target(a)
        sao_chep(a, b)  # Lưu dữ liệu ban đầu
        cap_nhap_vi_tri_kl(a[0], a[target], n)
        nho = a[target].f  # Biến "nho" sẽ lưu fitness cũ để so sánh

        choice = random.random()
        if choice < 0.3:  # TH target bị ẩn
            a[0].x = a[target].x.copy()
            best, second_best = tim_hai_con_tot_nhat(a)  # Lấy hai con tốt nhất
            a[target] = tao_con_moi_sau_khi_con_cu_chet(a, best, second_best, n)
        else:  # TH target chạy
            choice = random.random()
            if choice < 0.5:
                dich_chuyen_tuc_thoi(a[target], n)
            else:
                cap_nhap_vi_tri_tg(a[target], a[0], n)
        
        # Cập nhật vị trí và tính fitness cho các non-target
        for i in range(1, 11):
            h = random.random()
            if h <= 0.2:
                best, second_best = tim_hai_con_tot_nhat(a)  # Lấy hai con tốt nhất
                a[i] = tao_con_moi_sau_khi_con_cu_chet(a, best, second_best, n)  # Tạo con mới
            else:
                cap_nhap_vi_tri_ntg(a[i], n)
        
        # Tính Fitness 2 
        a[0].f2 = return_fitness2_kl(a[0], n)
        for i in range(1, 11):
            a[i].f2 = return_fitness2_tg(a[i], a[0], n)

        # Lưu lại giá trị best fitness sau mỗi vòng lặp
        best_fitness = min(prey.f for prey in a)
        best_fitness_history.append(best_fitness)

        # Vẽ lại đồ thị
        update_plot(best_fitness_history, ax)
        plt.pause(0.5)  # Tạm dừng để hiển thị đồ thị
        
        # Kiểm tra cải thiện fitness
        j = min(range(1, 11), key=lambda i: a[i].f2)
        if nho > a[j].f2:
            chuyen_doi_tu_f2_sang_f1(a)
            so_vong_khong_cai_thien = 0  # Reset đếm khi có cải thiện
            print(f"----------------------------Xong vòng {loop_n}----------------------------\n\n")
        else:
            print("Không cải thiện, làm lại bước này\n")
            sao_chep(b, a)
            so_vong_khong_cai_thien += 1  # Tăng đếm số vòng không cải thiện
            print(f"----------------------------Xong vòng {loop_n}----------------------------\n\n")
        
        # Nếu số vòng không cải thiện vượt quá giới hạn, loại bỏ 2 con xấu nhất và sinh 2 con mới
        if so_vong_khong_cai_thien >= gioi_han_khong_cai_thien:
            print("Số vòng không cải thiện đạt giới hạn, loại bỏ 2 con xấu nhất")
            worst1, worst2 = tim_hai_con_xau_nhat(a)
            best, second_best = tim_hai_con_tot_nhat(a)
            a[worst1] = sinh_con_moi_tu_hai_con_tot(a, best, second_best, n)
            a[worst2] = sinh_con_moi_tu_hai_con_tot(a, best, second_best, n)
            so_vong_khong_cai_thien = 0  # Reset đếm
        
    plt.show()

if __name__ == "__main__":
    main()

# Không có trường hợp loại bỏ 2 con fitness xấu nhất
"""def main():
    random.seed(42)
    a = [Prey() for _ in range(11)]
    b = [Prey() for _ in range(11)]
    target = 0

    print("Hãy chọn số chiều:")
    n = int(input())
    khoi_tao_vi_tri(a, n)
    print("Vị trí khởi tạo ban đầu là:")
    xuat(a)

    random.seed()  # Bỏ cố định seed    
    print("Hãy nhập số vòng bạn muốn thực hiện:")
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
        if choice < 0.3:  # TH target bị ẩn
            a[0].x = a[target].x.copy()
            best, second_best = tim_hai_con_tot_nhat(a)  # Lấy hai con tốt nhất
            a[target] = tao_con_moi_sau_khi_con_cu_chet(a, best, second_best, n)
            #print("Target bị ẩn")
        else:  # TH target chạy
            choice = random.random()
            if choice < 0.5:
                dich_chuyen_tuc_thoi(a[target], n)
            else:
                cap_nhap_vi_tri_tg(a[target], a[0], n)
        
        # Cập nhật vị trí và tính fitness cho các non-target
        for i in range(1, 11):
            h = random.random()
            if h <= 0.2:
                best, second_best = tim_hai_con_tot_nhat(a)  # Lấy hai con tốt nhất
                a[i] = tao_con_moi_sau_khi_con_cu_chet(a, best, second_best, n)  # Tạo con mới
                #print(f"Có non-target {i} bị trúng độc")
            else:
                cap_nhap_vi_tri_ntg(a[i], n)
        
        # Tính Fitness 2 
        a[0].f2 = return_fitness2_kl(a[0], n)
        for i in range(1, 11):
            a[i].f2 = return_fitness2_tg(a[i], a[0], n)

        # Lưu lại giá trị best fitness sau mỗi vòng lặp
        best_fitness = min(prey.f for prey in a)
        best_fitness_history.append(best_fitness)

        # Vẽ lại đồ thị
        update_plot(best_fitness_history, ax)
        plt.pause(0.5)  # Tạm dừng để hiển thị đồ thị
        
        # Kiểm tra cải thiện fitness
        j = min(range(1, 11), key=lambda i: a[i].f2)
        if nho > a[j].f2:
            chuyen_doi_tu_f2_sang_f1(a)
            print(f"----------------------------Xong vòng {loop_n}----------------------------\n\n")
        else:
            print("Không cải thiện, làm lại bước này\n")
            sao_chep(b, a)
            print(f"----------------------------Xong vòng {loop_n}----------------------------\n\n")
        #xuat(a)

    plt.show()

if __name__ == "__main__":
    main()"""
