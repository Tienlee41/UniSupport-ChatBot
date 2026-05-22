from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


SYSTEM_PROMPT = (
    "Bạn là một AI tư vấn tuyển sinh đại học chuyên nghiệp. Hãy trả lời chính xác "
    "theo thông tin tham khảo được cung cấp; nếu tài liệu không đủ dữ liệu thì nói rõ "
    "phần thiếu, không suy đoán."
)


@dataclass(frozen=True)
class School:
    key: str
    name: str
    acronym: str
    city: str
    kind: str
    managing_body: str
    website: str
    address: str
    tuition_base: float
    rank_bonus: float
    lecturer_count: int
    student_count: int
    founded: int
    campus_count: int


@dataclass(frozen=True)
class Major:
    name: str
    code: str
    group: str
    base_score: float
    credits: int
    duration: str
    careers: tuple[str, ...]


SCHOOLS: list[School] = [
    School("uet", "Trường Đại học Công nghệ, ĐHQG Hà Nội", "UET", "Hà Nội", "Công lập", "Đại học Quốc gia Hà Nội", "https://uet.vnu.edu.vn/", "144 Xuân Thủy, Cầu Giấy, Hà Nội", 34.0, 2.45, 310, 7600, 2004, 1),
    School("hust", "Đại học Bách khoa Hà Nội", "HUST", "Hà Nội", "Công lập tự chủ", "Bộ Giáo dục và Đào tạo", "https://hust.edu.vn/", "Số 1 Đại Cồ Việt, Hai Bà Trưng, Hà Nội", 33.5, 2.70, 1250, 36000, 1956, 1),
    School("ptit", "Học viện Công nghệ Bưu chính Viễn thông", "PTIT", "Hà Nội", "Công lập", "Bộ Thông tin và Truyền thông", "https://ptit.edu.vn/", "Km10 Nguyễn Trãi, Hà Đông, Hà Nội", 29.5, 2.15, 720, 21000, 1997, 2),
    School("hus", "Trường Đại học Khoa học Tự nhiên, ĐHQG Hà Nội", "HUS", "Hà Nội", "Công lập", "Đại học Quốc gia Hà Nội", "https://hus.vnu.edu.vn/", "334 Nguyễn Trãi, Thanh Xuân, Hà Nội", 28.0, 1.80, 640, 12000, 1956, 2),
    School("uit", "Trường Đại học Công nghệ Thông tin, ĐHQG TP.HCM", "UIT", "TP.HCM", "Công lập", "Đại học Quốc gia TP.HCM", "https://uit.edu.vn/", "Khu phố 6, Linh Trung, Thủ Đức, TP.HCM", 33.0, 2.55, 420, 11000, 2006, 1),
    School("hcmut", "Trường Đại học Bách khoa, ĐHQG TP.HCM", "HCMUT", "TP.HCM", "Công lập tự chủ", "Đại học Quốc gia TP.HCM", "https://hcmut.edu.vn/", "268 Lý Thường Kiệt, Quận 10, TP.HCM", 38.0, 2.50, 1150, 30000, 1957, 2),
    School("neu", "Đại học Kinh tế Quốc dân", "NEU", "Hà Nội", "Công lập tự chủ", "Bộ Giáo dục và Đào tạo", "https://neu.edu.vn/", "207 Giải Phóng, Hai Bà Trưng, Hà Nội", 24.5, 1.85, 850, 25000, 1956, 1),
    School("ftu", "Đại học Ngoại thương", "FTU", "Hà Nội", "Công lập tự chủ", "Bộ Giáo dục và Đào tạo", "https://ftu.edu.vn/", "91 Chùa Láng, Đống Đa, Hà Nội", 25.0, 2.00, 760, 22000, 1960, 3),
    School("utc", "Trường Đại học Giao thông Vận tải", "UTC", "Hà Nội", "Công lập", "Bộ Giáo dục và Đào tạo", "https://utc.edu.vn/", "Số 3 Cầu Giấy, Láng Thượng, Hà Nội", 22.0, 1.20, 760, 19000, 1945, 2),
    School("haui", "Trường Đại học Công nghiệp Hà Nội", "HAUI", "Hà Nội", "Công lập", "Bộ Công Thương", "https://haui.edu.vn/", "298 Cầu Diễn, Bắc Từ Liêm, Hà Nội", 23.5, 1.35, 980, 34000, 1898, 3),
    School("ulaw", "Trường Đại học Luật, ĐHQG Hà Nội", "VNU-UL", "Hà Nội", "Công lập", "Đại học Quốc gia Hà Nội", "https://law.vnu.edu.vn/", "144 Xuân Thủy, Cầu Giấy, Hà Nội", 26.0, 1.40, 180, 4300, 2022, 1),
    School("ulis", "Trường Đại học Ngoại ngữ, ĐHQG Hà Nội", "ULIS", "Hà Nội", "Công lập", "Đại học Quốc gia Hà Nội", "https://ulis.vnu.edu.vn/", "Phạm Văn Đồng, Cầu Giấy, Hà Nội", 23.0, 1.25, 540, 13000, 1955, 1),
    School("ussh", "Trường Đại học Khoa học Xã hội và Nhân văn, ĐHQG Hà Nội", "USSH", "Hà Nội", "Công lập", "Đại học Quốc gia Hà Nội", "https://ussh.vnu.edu.vn/", "336 Nguyễn Trãi, Thanh Xuân, Hà Nội", 22.5, 1.20, 560, 15000, 1945, 1),
    School("ueb", "Trường Đại học Kinh tế, ĐHQG Hà Nội", "UEB", "Hà Nội", "Công lập", "Đại học Quốc gia Hà Nội", "https://ueb.edu.vn/", "144 Xuân Thủy, Cầu Giấy, Hà Nội", 27.0, 1.65, 360, 9000, 1974, 1),
    School("tmU", "Trường Đại học Thương mại", "TMU", "Hà Nội", "Công lập", "Bộ Giáo dục và Đào tạo", "https://tmu.edu.vn/", "79 Hồ Tùng Mậu, Cầu Giấy, Hà Nội", 23.0, 1.25, 720, 24000, 1960, 1),
    School("act", "Học viện Kỹ thuật Mật mã", "KMA", "Hà Nội", "Công lập", "Ban Cơ yếu Chính phủ", "https://actvn.edu.vn/", "141 Chiến Thắng, Hà Đông, Hà Nội", 19.0, 1.45, 350, 8200, 1976, 2),
    School("ajc", "Học viện Báo chí và Tuyên truyền", "AJC", "Hà Nội", "Công lập", "Học viện Chính trị quốc gia Hồ Chí Minh", "https://ajc.hcma.vn/", "36 Xuân Thủy, Cầu Giấy, Hà Nội", 22.0, 1.15, 430, 10000, 1962, 1),
    School("hnue", "Trường Đại học Sư phạm Hà Nội", "HNUE", "Hà Nội", "Công lập", "Bộ Giáo dục và Đào tạo", "https://hnue.edu.vn/", "136 Xuân Thủy, Cầu Giấy, Hà Nội", 19.5, 1.30, 860, 18000, 1951, 1),
    School("cmc", "Trường Đại học CMC", "CMC", "Hà Nội", "Tư thục", "Tập đoàn CMC", "https://cmcu.edu.vn/", "11 Duy Tân, Cầu Giấy, Hà Nội", 38.0, 0.95, 240, 6000, 2022, 3),
    School("fpt", "Trường Đại học FPT", "FPTU", "Hà Nội", "Tư thục", "Tập đoàn FPT", "https://daihoc.fpt.edu.vn/", "Khu Công nghệ cao Hòa Lạc, Thạch Thất, Hà Nội", 55.0, 1.15, 900, 52000, 2006, 5),
    School("phenikaa", "Trường Đại học Phenikaa", "PU", "Hà Nội", "Tư thục", "Tập đoàn Phenikaa", "https://phenikaa-uni.edu.vn/", "Yên Nghĩa, Hà Đông, Hà Nội", 32.0, 1.05, 520, 17000, 2007, 1),
    School("hcmus", "Trường Đại học Khoa học Tự nhiên, ĐHQG TP.HCM", "HCMUS", "TP.HCM", "Công lập", "Đại học Quốc gia TP.HCM", "https://hcmus.edu.vn/", "227 Nguyễn Văn Cừ, Quận 5, TP.HCM", 29.0, 1.95, 760, 17000, 1941, 2),
    School("ueh", "Đại học Kinh tế TP.HCM", "UEH", "TP.HCM", "Công lập tự chủ", "Bộ Giáo dục và Đào tạo", "https://ueh.edu.vn/", "59C Nguyễn Đình Chiểu, Quận 3, TP.HCM", 28.0, 1.65, 980, 33000, 1976, 10),
    School("ctu", "Trường Đại học Cần Thơ", "CTU", "Cần Thơ", "Công lập", "Bộ Giáo dục và Đào tạo", "https://ctu.edu.vn/", "Khu II, đường 3/2, Ninh Kiều, Cần Thơ", 18.5, 1.00, 1200, 46000, 1966, 4),
]


MAJORS: list[Major] = [
    Major("Công nghệ thông tin", "7480201", "Công nghệ", 24.0, 140, "4 năm", ("lập trình viên", "kỹ sư hệ thống", "chuyên viên phân tích dữ liệu")),
    Major("Khoa học máy tính", "7480101", "Công nghệ", 24.8, 140, "4 năm", ("kỹ sư AI", "nhà khoa học dữ liệu", "kỹ sư phần mềm")),
    Major("Kỹ thuật phần mềm", "7480103", "Công nghệ", 24.5, 140, "4 năm", ("kỹ sư phần mềm", "kiểm thử phần mềm", "quản lý sản phẩm số")),
    Major("Trí tuệ nhân tạo", "7480207", "Công nghệ", 25.0, 140, "4 năm", ("kỹ sư AI", "kỹ sư thị giác máy tính", "chuyên viên MLOps")),
    Major("An toàn thông tin", "7480202", "Công nghệ", 23.8, 140, "4 năm", ("chuyên viên an ninh mạng", "kiểm thử xâm nhập", "quản trị bảo mật")),
    Major("Hệ thống thông tin", "7480104", "Công nghệ", 23.2, 135, "4 năm", ("phân tích nghiệp vụ", "quản trị hệ thống", "chuyên viên dữ liệu")),
    Major("Kỹ thuật máy tính", "7480106", "Công nghệ", 23.4, 145, "4 năm", ("kỹ sư nhúng", "kỹ sư IoT", "kỹ sư phần cứng")),
    Major("Điện tử viễn thông", "7520207", "Kỹ thuật", 22.5, 145, "4 năm", ("kỹ sư mạng", "kỹ sư viễn thông", "kỹ sư thiết kế mạch")),
    Major("Kỹ thuật điều khiển và tự động hóa", "7520216", "Kỹ thuật", 22.8, 145, "4 năm", ("kỹ sư tự động hóa", "kỹ sư robot", "kỹ sư vận hành")),
    Major("Logistics và Quản lý chuỗi cung ứng", "7510605", "Kinh tế", 23.3, 130, "4 năm", ("chuyên viên logistics", "điều phối vận tải", "quản lý chuỗi cung ứng")),
    Major("Kinh tế quốc tế", "7310106", "Kinh tế", 24.2, 130, "4 năm", ("chuyên viên xuất nhập khẩu", "chuyên viên phân tích thị trường", "tư vấn thương mại")),
    Major("Marketing", "7340115", "Kinh tế", 23.6, 130, "4 năm", ("chuyên viên marketing", "quản trị thương hiệu", "nghiên cứu thị trường")),
    Major("Quản trị kinh doanh", "7340101", "Kinh tế", 22.8, 130, "4 năm", ("chuyên viên kinh doanh", "quản lý dự án", "khởi nghiệp")),
    Major("Kế toán", "7340301", "Kinh tế", 21.8, 130, "4 năm", ("kế toán viên", "kiểm toán viên", "chuyên viên tài chính")),
    Major("Ngôn ngữ Anh", "7220201", "Ngôn ngữ", 22.0, 130, "4 năm", ("biên phiên dịch", "giảng dạy ngoại ngữ", "truyền thông quốc tế")),
    Major("Luật", "7380101", "Luật", 22.4, 125, "4 năm", ("chuyên viên pháp chế", "luật sư tập sự", "tư vấn pháp luật")),
    Major("Sư phạm Toán học", "7140209", "Sư phạm", 22.6, 130, "4 năm", ("giáo viên toán", "chuyên viên giáo dục", "nghiên cứu giáo dục")),
    Major("Công nghệ sinh học", "7420201", "Khoa học tự nhiên", 21.6, 135, "4 năm", ("kỹ thuật viên phòng thí nghiệm", "nghiên cứu viên", "chuyên viên kiểm nghiệm")),
    Major("Xã hội học", "7310301", "Xã hội", 20.5, 125, "4 năm", ("nghiên cứu xã hội", "truyền thông cộng đồng", "quản lý dự án xã hội")),
    Major("Y khoa", "7720101", "Sức khỏe", 26.0, 170, "6 năm", ("bác sĩ đa khoa", "nghiên cứu y sinh", "quản lý y tế")),
]

MAJOR_BY_NAME = {major.name: major for major in MAJORS}


METHODS = ["THPT", "Học bạ", "ĐGNL", "ĐGTD", "HSA", "TSA"]
PROGRAMS = ["Đại trà", "Chất lượng cao", "Liên kết quốc tế", "Định hướng nghề nghiệp"]
YEARS = [2022, 2023, 2024, 2025, 2026]
SEMESTERS = ["kỳ 1", "kỳ 2", "học kỳ hè"]


def stable_int(*parts: object) -> int:
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16)


def pick(rng: random.Random, values: list | tuple):
    return values[rng.randrange(len(values))]


def offered_majors(school: School) -> list[Major]:
    technology = {"Công nghệ"}
    engineering = {"Kỹ thuật"}
    econ = {"Kinh tế"}
    natural = {"Khoa học tự nhiên"}
    language = {"Ngôn ngữ"}
    law = {"Luật"}
    social = {"Xã hội"}
    pedagogy = {"Sư phạm"}
    health = {"Sức khỏe"}
    groups_by_school = {
        "uet": technology | engineering,
        "hust": technology | engineering | econ | natural,
        "ptit": technology | engineering | econ,
        "hus": technology | natural,
        "uit": technology,
        "hcmut": technology | engineering | natural,
        "neu": econ | technology,
        "ftu": econ | language,
        "utc": technology | engineering | econ,
        "haui": technology | engineering | econ | language,
        "ulaw": law,
        "ulis": language,
        "ussh": social | language,
        "ueb": econ | technology,
        "tmU": econ,
        "act": technology,
        "ajc": social | language,
        "hnue": pedagogy | language | natural | technology,
        "cmc": technology | econ | language,
        "fpt": technology | econ | language,
        "phenikaa": technology | engineering | econ | health,
        "hcmus": technology | natural,
        "ueh": econ | technology,
        "ctu": technology | engineering | econ | natural | language | law | pedagogy,
    }
    groups = groups_by_school.get(school.key, technology | econ)
    return [major for major in MAJORS if major.group in groups]


def schools_for_major(major: Major, city: str | None = None) -> list[School]:
    schools = [school for school in SCHOOLS if major in offered_majors(school)]
    if city:
        schools = [school for school in schools if school.city == city]
    return schools


def score_for(school: School, major: Major, year: int, method: str = "THPT", program: str = "Đại trà") -> float:
    method_adj = {"THPT": 0.0, "Học bạ": 0.6, "ĐGNL": 1.1, "ĐGTD": 0.9, "HSA": 1.0, "TSA": 0.8}.get(method, 0.0)
    program_adj = {"Đại trà": 0.0, "Chất lượng cao": 0.25, "Liên kết quốc tế": -0.35, "Định hướng nghề nghiệp": -0.15}.get(program, 0.0)
    year_adj = (year - 2022) * 0.18
    noise = (stable_int(school.key, major.code, year, method, program) % 101) / 100 - 0.50
    value = major.base_score + school.rank_bonus + method_adj + program_adj + year_adj + noise
    if method in {"ĐGNL", "HSA", "TSA", "ĐGTD"}:
        return round(min(max(value, 18.0), 30.0), 2)
    return round(min(max(value, 15.0), 29.95), 2)


def floor_for(school: School, major: Major, year: int) -> float:
    gap = 1.2 + (stable_int("floor", school.key, major.code, year) % 14) / 10
    return round(max(15.0, min(score_for(school, major, year) - gap, 24.5)), 2)


def tuition_for(school: School, major: Major, year: int, program: str = "Đại trà") -> float:
    group_adj = {"Công nghệ": 5.0, "Kỹ thuật": 3.0, "Sức khỏe": 7.0, "Kinh tế": 2.0, "Ngôn ngữ": 1.0}.get(major.group, 0.0)
    program_mult = {"Đại trà": 1.0, "Chất lượng cao": 1.45, "Liên kết quốc tế": 2.15, "Định hướng nghề nghiệp": 1.2}.get(program, 1.0)
    year_adj = (year - 2022) * 1.6
    noise = (stable_int("tuition", school.key, major.code, year, program) % 40) / 10
    return round((school.tuition_base + group_adj + year_adj + noise) * program_mult, 1)


def quota_for(school: School, major: Major, year: int) -> int:
    base = 60 + stable_int("quota", school.key, major.code, year) % 420
    if major.group == "Công nghệ":
        base += 90
    if school.student_count > 25000:
        base += 80
    return int(round(base / 5) * 5)


def combo_for(major: Major) -> str:
    if major.group in {"Công nghệ", "Kỹ thuật"}:
        return "A00, A01, D01"
    if major.group == "Kinh tế":
        return "A00, A01, D01, D07"
    if major.group == "Ngôn ngữ":
        return "D01, D14, D15"
    if major.group == "Sức khỏe":
        return "B00, A00"
    return "C00, D01, A01"


def relevant_majors(seed: int, include: Major | None = None, count: int = 6, school: School | None = None) -> list[Major]:
    rng = random.Random(seed)
    majors = offered_majors(school) if school else MAJORS[:]
    rng.shuffle(majors)
    selected = majors[:count]
    if include and include not in selected:
        if include in majors:
            selected[-1] = include
        else:
            selected.append(include)
    return selected


def format_money(value: float) -> str:
    return f"{value:.1f} triệu đồng/năm".replace(".0", "")


def doc_admission(doc_id: str, school: School, year: int, majors: list[Major], method: str = "THPT") -> dict:
    rows = []
    for major in majors:
        rows.append(
            f"| {major.code} | {major.name} | {combo_for(major)} | {quota_for(school, major, year)} | "
            f"{floor_for(school, major, year)} | {score_for(school, major, year, method)} |"
        )
    content = (
        f"# Điểm chuẩn tuyển sinh đại học chính quy {year} - {school.name}\n"
        f"Phương thức xét tuyển: {method}. Điểm chuẩn là điểm trúng tuyển, khác với điểm sàn/điểm nhận hồ sơ.\n\n"
        "| Mã ngành | Ngành | Tổ hợp xét tuyển | Chỉ tiêu | Điểm sàn | Điểm chuẩn |\n"
        "|---|---|---|---:|---:|---:|\n"
        + "\n".join(rows)
    )
    return {"doc_id": doc_id, "title": f"Điểm chuẩn {school.acronym} {year}", "source": f"synthetic://admission/{school.key}/{year}", "content": content, "is_relevant": True}


def doc_floor_only(doc_id: str, school: School, year: int, majors: list[Major]) -> dict:
    rows = [f"| {major.code} | {major.name} | {combo_for(major)} | {floor_for(school, major, year)} |" for major in majors]
    content = (
        f"# Ngưỡng đảm bảo chất lượng đầu vào {year} - {school.name}\n"
        "Tài liệu này chỉ công bố điểm sàn/điểm nhận hồ sơ, chưa công bố điểm chuẩn trúng tuyển.\n\n"
        "| Mã ngành | Ngành | Tổ hợp | Điểm sàn |\n|---|---|---|---:|\n"
        + "\n".join(rows)
    )
    return {"doc_id": doc_id, "title": f"Điểm sàn {school.acronym} {year}", "source": f"synthetic://floor/{school.key}/{year}", "content": content, "is_relevant": True}


def doc_tuition(doc_id: str, school: School, year: int, majors: list[Major]) -> dict:
    rows = []
    for major in majors:
        rows.append(
            f"| {major.name} | Đại trà | {format_money(tuition_for(school, major, year, 'Đại trà'))} | "
            f"Chất lượng cao | {format_money(tuition_for(school, major, year, 'Chất lượng cao'))} |"
        )
    content = (
        f"# Học phí dự kiến năm học {year}-{year + 1} - {school.name}\n"
        "Đơn vị tính: triệu đồng/sinh viên/năm học. Mức thu có thể điều chỉnh theo số tín chỉ đăng ký thực tế.\n\n"
        "| Ngành | Chương trình | Học phí | Chương trình mở rộng | Học phí |\n|---|---|---:|---|---:|\n"
        + "\n".join(rows)
    )
    return {"doc_id": doc_id, "title": f"Học phí {school.acronym} {year}", "source": f"synthetic://tuition/{school.key}/{year}", "content": content, "is_relevant": True}


def doc_public(doc_id: str, school: School, year: int) -> dict:
    gs_pgs = max(4, int(school.lecturer_count * (0.06 + school.rank_bonus / 100)))
    phd = int(school.lecturer_count * (0.43 + min(school.rank_bonus, 2.7) / 10))
    master = max(0, school.lecturer_count - phd - gs_pgs)
    library_books = 85000 + stable_int("books", school.key) % 420000
    classrooms = 55 + stable_int("rooms", school.key) % 180
    labs = 15 + stable_int("labs", school.key) % 85
    content = (
        f"# Báo cáo công khai năm {year} - {school.name}\n"
        f"Tên viết tắt: {school.acronym}. Loại hình: {school.kind}. Cơ quan chủ quản: {school.managing_body}.\n"
        f"Địa chỉ chính: {school.address}. Website: {school.website}. Số cơ sở/campus: {school.campus_count}.\n"
        f"Quy mô người học: khoảng {school.student_count:,} sinh viên. Giảng viên cơ hữu: {school.lecturer_count} người.\n"
        f"Trong đó có khoảng {gs_pgs} GS/PGS, {phd} tiến sĩ và {master} thạc sĩ/cử nhân tham gia giảng dạy.\n"
        f"Cơ sở vật chất công khai gồm {classrooms} phòng học, {labs} phòng thí nghiệm/thực hành và thư viện khoảng {library_books:,} đầu sách/tài liệu."
    )
    return {"doc_id": doc_id, "title": f"Công khai {school.acronym} {year}", "source": f"synthetic://public/{school.key}/{year}", "content": content, "is_relevant": True}


def doc_program(doc_id: str, school: School, major: Major, year: int) -> dict:
    courses = {
        "Công nghệ": ["Nhập môn lập trình", "Cấu trúc dữ liệu", "Cơ sở dữ liệu", "Mạng máy tính", "Đồ án tốt nghiệp"],
        "Kỹ thuật": ["Toán kỹ thuật", "Mạch điện", "Điều khiển tự động", "Thiết kế hệ thống", "Đồ án kỹ thuật"],
        "Kinh tế": ["Kinh tế vi mô", "Quản trị học", "Phân tích dữ liệu kinh doanh", "Marketing căn bản", "Thực tập tốt nghiệp"],
        "Ngôn ngữ": ["Ngữ âm", "Biên dịch", "Phiên dịch", "Văn hóa học", "Thực tập nghề nghiệp"],
        "Luật": ["Lý luận nhà nước pháp luật", "Luật dân sự", "Luật hình sự", "Luật thương mại", "Khóa luận"],
    }.get(major.group, ["Nhập môn ngành", "Phương pháp nghiên cứu", "Học phần chuyên ngành", "Thực tập", "Khóa luận"])
    content = (
        f"# Chương trình đào tạo ngành {major.name} - {school.name} ({year})\n"
        f"Mã ngành: {major.code}. Thời gian đào tạo: {major.duration}. Khối lượng kiến thức toàn khóa: {major.credits} tín chỉ.\n"
        f"Tổ hợp xét tuyển thường dùng: {combo_for(major)}. Chương trình có các học phần tiêu biểu: {', '.join(courses)}.\n"
        f"Chuẩn đầu ra nhấn mạnh năng lực chuyên môn, ngoại ngữ bậc 3/6 hoặc tương đương, kỹ năng làm việc nhóm và đạo đức nghề nghiệp.\n"
        f"Vị trí việc làm phù hợp: {', '.join(major.careers)}."
    )
    return {"doc_id": doc_id, "title": f"CTĐT {major.name} {school.acronym}", "source": f"synthetic://program/{school.key}/{major.code}/{year}", "content": content, "is_relevant": True}


def doc_academic(doc_id: str, school: School, year: int, semester: str) -> dict:
    start_day = 4 + stable_int("start", school.key, year, semester) % 12
    end_day = start_day + 10
    payment_day = end_day + 7
    content = (
        f"# Thông báo học vụ {semester} năm học {year}-{year + 1} - {school.name}\n"
        f"Đăng ký học phần trực tuyến từ ngày {start_day:02d}/08/{year} đến {end_day:02d}/08/{year} trên cổng đào tạo của trường.\n"
        f"Sinh viên học lại hoặc học cải thiện chọn đúng mã học phần, nhóm lớp và xác nhận trước khi hết hạn đăng ký.\n"
        f"Hạn cuối nộp học phí {semester}: {payment_day:02d}/09/{year}. Kênh thanh toán: cổng sinh viên, chuyển khoản theo mã định danh hoặc quầy tài chính.\n"
        "Nếu nợ học phí quá hạn, sinh viên có thể bị khóa đăng ký học phần kỳ tiếp theo cho đến khi hoàn tất nghĩa vụ tài chính."
    )
    return {"doc_id": doc_id, "title": f"Học vụ {school.acronym} {semester} {year}", "source": f"synthetic://academic/{school.key}/{year}/{semester}", "content": content, "is_relevant": True}


def doc_accreditation(doc_id: str, school: School, major: Major, year: int) -> dict:
    org = pick(random.Random(stable_int("org", school.key, major.code)), ["CEA-UD", "VNU-CEA", "AUN-QA", "FIBAA", "HCERES"])
    start = year - (stable_int("acc", school.key, major.code) % 4)
    expiry = start + 5
    level = pick(random.Random(stable_int("level", school.key, major.code)), ["đạt chuẩn chất lượng", "đạt mức 4/7", "đạt chuẩn khu vực", "đang trong chu kỳ cải tiến sau đánh giá"])
    content = (
        f"# Kiểm định chất lượng - {school.name}\n"
        f"Cơ sở giáo dục {school.acronym} được đánh giá ngoài trong chu kỳ {start}-{expiry} bởi {org} và kết luận {level}.\n"
        f"Chương trình đào tạo ngành {major.name} ({major.code}) có báo cáo tự đánh giá, minh chứng đội ngũ giảng viên, cơ sở vật chất, "
        f"chuẩn đầu ra và khảo sát việc làm người học. Giấy chứng nhận có hiệu lực đến hết năm {expiry} nếu không có thay đổi bất thường."
    )
    return {"doc_id": doc_id, "title": f"Kiểm định {school.acronym} {major.name}", "source": f"synthetic://accreditation/{school.key}/{major.code}", "content": content, "is_relevant": True}


def doc_ranking(doc_id: str, schools: list[School], major: Major, year: int, method: str = "THPT") -> dict:
    rows = []
    for school in schools:
        tuition = tuition_for(school, major, year)
        rows.append(
            f"| {school.name} | {school.acronym} | {combo_for(major)} | {score_for(school, major, year, method)} | {format_money(tuition)} |"
        )
    content = (
        f"# Bảng tổng hợp điểm chuẩn và học phí ngành {major.name} năm {year}\n"
        f"Bảng chỉ dùng điểm chuẩn phương thức {method}; không sử dụng điểm sàn để xếp hạng.\n\n"
        "| Trường | Viết tắt | Tổ hợp | Điểm chuẩn | Học phí dự kiến |\n|---|---|---|---:|---:|\n"
        + "\n".join(rows)
    )
    return {"doc_id": doc_id, "title": f"Tổng hợp {major.name} {year}", "source": f"synthetic://ranking/{major.code}/{year}", "content": content, "is_relevant": True}


def distractor_doc(doc_id: str, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    year = pick(rng, YEARS)
    topic = pick(rng, ["du học Mỹ", "điểm sàn", "tin tuyển dụng", "cao học", "hoạt động đoàn"])
    content = (
        f"# Bản tin {topic} {year}\n"
        f"Tài liệu này nói về {topic} tại {school.name}, không cung cấp trực tiếp dữ liệu điểm chuẩn, học phí hoặc kiểm định theo câu hỏi chính."
    )
    return {"doc_id": doc_id, "title": f"Nhiễu - {topic}", "source": f"synthetic://distractor/{topic}/{school.key}/{year}", "content": content, "is_relevant": False}


def cite(doc: dict) -> str:
    return f"[{doc['doc_id']}]"


def answer_header(title: str, bullets: list[str], table: str | None, note: str, sources: list[dict]) -> str:
    body = [f"### {title}", "**Tóm tắt:**"]
    body.extend(f"- {item}" for item in bullets)
    if table:
        body.append("")
        body.append(table)
    body.append("")
    body.append("**Nguồn:** " + ", ".join(cite(doc) for doc in sources))
    body.append(f"**Lưu ý:** {note}")
    return "\n".join(body)


def make_messages(question: str, docs: list[dict], answer: str) -> list[dict]:
    context_parts = []
    for index, doc in enumerate(docs, 1):
        context_parts.append(
            f"[{doc['doc_id']}] {doc['title']}\nNguồn: {doc['source']}\n{doc['content']}"
        )
    user = "Thông tin tham khảo:\n" + "\n\n---\n\n".join(context_parts) + f"\n\nCâu hỏi: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": answer},
    ]


def build_record(record_id: int, category: str, task_type: str, question: str, docs: list[dict], answer: str, metadata: dict) -> dict:
    split = "train" if record_id <= 4500 else ("validation" if record_id <= 4750 else "test")
    return {
        "id": f"lora_raft_{record_id:05d}",
        "split": split,
        "category": category,
        "task_type": task_type,
        "question": question,
        "answer": answer,
        "documents": docs,
        "messages": make_messages(question, docs, answer),
        "metadata": metadata,
    }


def gen_score(record_id: int, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    major = pick(rng, offered_majors(school))
    year = pick(rng, YEARS)
    method = pick(rng, METHODS[:4])
    program = pick(rng, PROGRAMS)
    docs = [doc_admission(f"D{record_id}_A", school, year, relevant_majors(record_id, major, school=school), method)]
    variant = record_id % 7
    if variant == 0:
        question = f"Điểm chuẩn ngành {major.name} trường {school.name} năm {year} là bao nhiêu?"
    elif variant == 1:
        question = f"Điểm chuẩn ngành {major.name} trường {school.acronym} theo phương thức {method} năm {year}?"
    elif variant == 2:
        question = f"Điểm sàn ngành {major.name} trường {school.acronym} năm {year}?"
    elif variant == 3:
        question = f"Điểm trúng tuyển ngành {major.name} của {school.name} năm {year} có cao hơn điểm sàn không?"
    elif variant == 4:
        question = f"Ngành nào có điểm chuẩn cao nhất trường {school.acronym} năm {year}?"
    elif variant == 5:
        question = f"Điểm chuẩn ngành {major.name} trường {school.acronym} chương trình {program} năm {year}?"
    else:
        question = f"Điểm chuẩn ngành {major.name} trường {school.acronym} năm {year} tăng hay giảm so với năm {year - 1}?"
        docs.append(doc_admission(f"D{record_id}_B", school, year - 1, relevant_majors(record_id + 11, major, school=school), method))

    if "Điểm sàn" in question:
        value = floor_for(school, major, year)
        bullets = [f"Điểm sàn ngành {major.name} là **{value}**.", "Đây là ngưỡng nhận hồ sơ, không phải điểm chuẩn trúng tuyển."]
        table = f"| Ngành | Năm | Điểm sàn |\n|---|---:|---:|\n| {major.name} | {year} | {value} |"
    elif "cao nhất" in question:
        candidates = relevant_majors(record_id, count=8, school=school)
        best = max(candidates, key=lambda m: score_for(school, m, year, method))
        value = score_for(school, best, year, method)
        bullets = [f"Ngành có điểm chuẩn cao nhất trong tài liệu là **{best.name}** với **{value}** điểm.", f"Phạm vi: phương thức {method}, năm {year}."]
        rows = sorted(candidates, key=lambda m: score_for(school, m, year, method), reverse=True)[:5]
        table = "| Ngành | Điểm chuẩn |\n|---|---:|\n" + "\n".join(f"| {m.name} | {score_for(school, m, year, method)} |" for m in rows)
    elif "tăng hay giảm" in question:
        now = score_for(school, major, year, method)
        prev = score_for(school, major, year - 1, method)
        delta = round(now - prev, 2)
        trend = "tăng" if delta > 0 else ("giảm" if delta < 0 else "không đổi")
        bullets = [f"Năm {year}: **{now}** điểm; năm {year - 1}: **{prev}** điểm.", f"Chênh lệch **{delta:+.2f}** điểm, tức là {trend}."]
        table = f"| Năm | Điểm chuẩn |\n|---:|---:|\n| {year - 1} | {prev} |\n| {year} | {now} |"
    else:
        value = score_for(school, major, year, method, program if "chương trình" in question else "Đại trà")
        floor = floor_for(school, major, year)
        bullets = [f"Điểm chuẩn ngành {major.name} là **{value}** theo phương thức {method}.", f"Điểm sàn cùng ngành trong tài liệu là **{floor}**, thấp hơn điểm chuẩn."]
        table = f"| Trường | Ngành | Phương thức | Điểm sàn | Điểm chuẩn |\n|---|---|---|---:|---:|\n| {school.acronym} | {major.name} | {method} | {floor} | {value} |"
    answer = answer_header(f"Điểm tuyển sinh {year} - {major.name} ({school.acronym})", bullets, table, "Chỉ dùng số liệu trong tài liệu; không trộn điểm sàn với điểm chuẩn.", docs)
    return build_record(record_id, "tuyen_sinh_diem", "qa_numeric", question, maybe_add_distractor(docs, rng, record_id), answer, {"school": school.acronym, "major": major.name, "year": year, "method": method})


def gen_quota_method(record_id: int, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    major = pick(rng, offered_majors(school))
    year = pick(rng, YEARS)
    method = pick(rng, METHODS)
    docs = [doc_admission(f"D{record_id}_A", school, year, relevant_majors(record_id, major, 7, school), method)]
    variant = record_id % 6
    if variant == 0:
        question = f"Chỉ tiêu tuyển sinh ngành {major.name} trường {school.name} năm {year} là bao nhiêu?"
    elif variant == 1:
        question = f"Tổng chỉ tiêu tuyển sinh trường {school.acronym} năm {year} trong các ngành ở tài liệu là bao nhiêu?"
    elif variant == 2:
        question = f"Ngành {major.name} trường {school.acronym} xét tuyển theo tổ hợp nào?"
    elif variant == 3:
        question = f"Trường {school.acronym} có sử dụng phương thức {method} năm {year} không?"
    elif variant == 4:
        question = f"Phương thức xét tuyển ngành {major.name} trường {school.acronym} năm {year} là gì?"
    else:
        question = f"Mã ngành và chỉ tiêu ngành {major.name} của {school.acronym} năm {year}?"
    quota = quota_for(school, major, year)
    total = sum(quota_for(school, m, year) for m in relevant_majors(record_id, major, 7, school))
    bullets = [
        f"Ngành {major.name} có mã ngành **{major.code}** và chỉ tiêu **{quota}**.",
        f"Tài liệu công bố phương thức **{method}** và tổ hợp xét tuyển **{combo_for(major)}**.",
        f"Tổng chỉ tiêu của các ngành xuất hiện trong tài liệu là **{total}**.",
    ]
    table = f"| Ngành | Mã ngành | Tổ hợp | Phương thức | Chỉ tiêu |\n|---|---|---|---|---:|\n| {major.name} | {major.code} | {combo_for(major)} | {method} | {quota} |"
    answer = answer_header(f"Chỉ tiêu và phương thức {year} - {school.acronym}", bullets, table, "Nếu cần tổng chỉ tiêu toàn trường, cần tài liệu đầy đủ tất cả ngành.", docs)
    return build_record(record_id, "tuyen_sinh_chi_tieu_phuong_thuc", "qa_fact", question, maybe_add_distractor(docs, rng, record_id), answer, {"school": school.acronym, "major": major.name, "year": year})


def gen_tuition(record_id: int, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    major = pick(rng, offered_majors(school))
    year = pick(rng, YEARS)
    docs = [doc_tuition(f"D{record_id}_T", school, year, relevant_majors(record_id, major, 6, school))]
    variant = record_id % 7
    if variant == 0:
        question = f"Học phí ngành {major.name} trường {school.name} năm {year} là bao nhiêu?"
    elif variant == 1:
        question = f"Chương trình Chất lượng cao ngành {major.name} tại {school.acronym} năm {year} học phí thế nào?"
    elif variant == 2:
        question = f"So sánh học phí đại trà và chất lượng cao ngành {major.name} của {school.acronym} năm {year}?"
    elif variant == 3:
        question = f"Học phí trường {school.acronym} năm {year} có mức nào cho ngành {major.name}?"
    elif variant == 4:
        question = f"Các khoản học phí dự kiến của {school.name} được tính theo đơn vị nào?"
    elif variant == 5:
        question = f"Học phí ngành {major.name} của {school.acronym} có dưới 50 triệu đồng/năm không?"
    else:
        question = f"Cho tôi biết học phí đại trà ngành {major.name} ở {school.acronym} năm học {year}-{year + 1}."
    base = tuition_for(school, major, year, "Đại trà")
    clc = tuition_for(school, major, year, "Chất lượng cao")
    below = "có" if base < 50 else "không"
    bullets = [
        f"Học phí đại trà ngành {major.name}: **{format_money(base)}**.",
        f"Học phí chất lượng cao cùng ngành: **{format_money(clc)}**.",
        f"Mức đại trà {below} dưới 50 triệu đồng/năm.",
    ]
    table = f"| Ngành | Đại trà | Chất lượng cao |\n|---|---:|---:|\n| {major.name} | {format_money(base)} | {format_money(clc)} |"
    answer = answer_header(f"Học phí {year}-{year + 1} - {major.name} ({school.acronym})", bullets, table, "Mức học phí là dự kiến theo năm học; học phí thực nộp phụ thuộc số tín chỉ.", docs)
    return build_record(record_id, "hoc_phi_tai_chinh", "qa_numeric", question, maybe_add_distractor(docs, rng, record_id), answer, {"school": school.acronym, "major": major.name, "year": year})


def gen_academic(record_id: int, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    year = pick(rng, YEARS)
    semester = pick(rng, SEMESTERS)
    docs = [doc_academic(f"D{record_id}_H", school, year, semester)]
    course = pick(rng, ["Cấu trúc dữ liệu", "Giải tích 1", "Kinh tế vi mô", "Tiếng Anh B1", "Triết học Mác-Lênin"])
    variant = record_id % 8
    questions = [
        f"Làm thế nào để đăng ký học phần tại trường {school.acronym}?",
        f"Thủ tục đăng ký học phần trường {school.name} {semester} năm học {year}-{year + 1}?",
        f"Hạn đăng ký học phần trường {school.acronym} {semester} năm học {year}-{year + 1}?",
        f"Làm thế nào để đăng ký học lại học phần {course} tại {school.acronym}?",
        f"Học phí {semester} năm học {year}-{year + 1} của sinh viên {school.acronym} đóng ở đâu?",
        f"Hạn cuối đóng học phí {semester} năm học {year}-{year + 1} trường {school.acronym} là khi nào?",
        f"Nợ học phí trường {school.acronym} xử lý như thế nào?",
        f"Cách thanh toán học phí {semester} của {school.name}?",
    ]
    question = questions[variant]
    start = re.search(r"từ ngày (\d{2}/08/\d{4})", docs[0]["content"]).group(1)
    end = re.search(r"đến (\d{2}/08/\d{4})", docs[0]["content"]).group(1)
    deadline = re.search(r"Hạn cuối nộp học phí.*?: (\d{2}/09/\d{4})", docs[0]["content"]).group(1)
    bullets = [
        f"Đăng ký học phần trực tuyến từ **{start}** đến **{end}**.",
        f"Hạn cuối nộp học phí {semester}: **{deadline}**.",
        "Có thể thanh toán qua cổng sinh viên, chuyển khoản theo mã định danh hoặc quầy tài chính.",
    ]
    table = f"| Nội dung | Thời hạn/Kênh thực hiện |\n|---|---|\n| Đăng ký học phần | {start} - {end} |\n| Nộp học phí | {deadline} |\n| Kênh thanh toán | Cổng sinh viên, chuyển khoản, quầy tài chính |"
    answer = answer_header(f"Học vụ {semester} {year}-{year + 1} - {school.acronym}", bullets, table, "Sinh viên cần kiểm tra đúng mã học phần và hoàn tất xác nhận trước hạn.", docs)
    return build_record(record_id, "hoc_vu_sinh_vien", "procedural", question, maybe_add_distractor(docs, rng, record_id), answer, {"school": school.acronym, "semester": semester, "year": year})


def gen_public(record_id: int, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    year = pick(rng, YEARS)
    major = pick(rng, offered_majors(school))
    public_doc = doc_public(f"D{record_id}_P", school, year)
    acc_doc = doc_accreditation(f"D{record_id}_K", school, major, year)
    docs = [public_doc, acc_doc] if record_id % 3 == 0 else [public_doc]
    variant = record_id % 12
    questions = [
        f"Theo báo cáo năm {year}, tên đầy đủ và tên viết tắt của trường {school.acronym} là gì?",
        f"Theo công khai năm {year}, trường {school.name} thuộc loại hình nào?",
        f"Trường {school.acronym} thuộc cơ quan chủ quản nào trong báo cáo năm {year}?",
        f"Năm {year}, trường {school.acronym} có bao nhiêu cơ sở/campus?",
        f"Địa chỉ đầy đủ của {school.name} theo báo cáo công khai năm {year} là gì?",
        f"Năm {year}, trường {school.acronym} có bao nhiêu giảng viên cơ hữu?",
        f"Số lượng giáo sư, phó giáo sư, tiến sĩ tại {school.acronym} năm {year}?",
        f"Thư viện của trường {school.acronym} có bao nhiêu đầu sách, tài liệu theo báo cáo năm {year}?",
        f"Năm {year}, trường {school.acronym} có bao nhiêu phòng học và phòng thí nghiệm?",
        f"Trường {school.acronym} đã được kiểm định cơ sở giáo dục chưa theo tài liệu năm {year}?",
        f"Giấy chứng nhận kiểm định ngành {major.name} của {school.acronym} còn hiệu lực đến năm nào trong tài liệu {year}?",
        f"Năm {year}, trường {school.acronym} có công khai danh sách và trình độ giảng viên không?",
    ]
    question = questions[variant]
    content = "\n".join(doc["content"] for doc in docs)
    bullets = []
    if "kiểm định" in question.lower():
        expiry_match = re.search(r"hiệu lực đến hết năm (\d{4})", content)
        org_match = re.search(r"bởi ([A-Z-]+)", content)
        bullets = [
            f"Tài liệu có thông tin kiểm định với tổ chức **{org_match.group(1) if org_match else 'không nêu rõ'}**.",
            f"Hiệu lực chứng nhận đến **{expiry_match.group(1) if expiry_match else 'không nêu rõ'}**.",
            f"Phạm vi liên quan đến ngành {major.name} và/hoặc cơ sở giáo dục.",
        ]
        table = f"| Nội dung | Thông tin |\n|---|---|\n| Trường | {school.name} |\n| Ngành | {major.name} |\n| Hiệu lực | {expiry_match.group(1) if expiry_match else 'không nêu rõ'} |"
    else:
        bullets = [
            f"Tên đầy đủ: **{school.name}**; viết tắt: **{school.acronym}**.",
            f"Loại hình: **{school.kind}**; cơ quan chủ quản: **{school.managing_body}**.",
            f"Địa chỉ chính: **{school.address}**; quy mô khoảng **{school.student_count:,}** sinh viên.",
        ]
        table = f"| Mục | Thông tin |\n|---|---|\n| Tên trường | {school.name} |\n| Viết tắt | {school.acronym} |\n| Loại hình | {school.kind} |\n| Cơ quan chủ quản | {school.managing_body} |\n| Địa chỉ | {school.address} |"
    answer = answer_header(f"Thông tin công khai {year} - {school.acronym}", bullets, table, "Các số liệu công khai có thể được cập nhật theo từng năm báo cáo.", docs)
    return build_record(record_id, "cong_khai_kiem_dinh", "qa_fact", question, maybe_add_distractor(docs, rng, record_id), answer, {"school": school.acronym, "year": year, "major": major.name})


def gen_program(record_id: int, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    major = pick(rng, offered_majors(school))
    year = pick(rng, YEARS)
    docs = [doc_program(f"D{record_id}_C", school, major, year), doc_accreditation(f"D{record_id}_K", school, major, year)]
    variant = record_id % 9
    questions = [
        f"Trường {school.acronym} có đào tạo ngành {major.name} không?",
        f"Thời gian đào tạo ngành {major.name} trường {school.name}?",
        f"Số tín chỉ ngành {major.name} trường {school.acronym}?",
        f"Chương trình đào tạo ngành {major.name} trường {school.acronym} gồm gì?",
        f"Danh sách học phần tiêu biểu ngành {major.name} ở {school.acronym}?",
        f"Chuẩn đầu ra ngành {major.name} của trường {school.acronym}?",
        f"Ngành {major.name} ra trường làm vị trí nào?",
        f"Ngành {major.name} trường {school.acronym} đã được kiểm định chưa?",
        f"Cơ hội việc làm ngành {major.name} tại {school.acronym} như thế nào?",
    ]
    question = questions[variant]
    bullets = [
        f"Ngành {major.name} có mã **{major.code}**, thời gian đào tạo **{major.duration}**.",
        f"Khối lượng toàn khóa là **{major.credits} tín chỉ**.",
        f"Vị trí việc làm phù hợp: **{', '.join(major.careers)}**.",
    ]
    table = f"| Nội dung | Thông tin |\n|---|---|\n| Ngành | {major.name} |\n| Mã ngành | {major.code} |\n| Thời gian | {major.duration} |\n| Tín chỉ | {major.credits} |\n| Việc làm | {', '.join(major.careers)} |"
    answer = answer_header(f"Ngành {major.name} - {school.acronym}", bullets, table, "Thông tin chương trình cần đối chiếu với phiên bản CTĐT áp dụng cho khóa tuyển sinh cụ thể.", docs)
    return build_record(record_id, "nganh_dao_tao_nghe_nghiep", "qa_fact", question, maybe_add_distractor(docs, rng, record_id), answer, {"school": school.acronym, "major": major.name, "year": year})


def gen_comparison(record_id: int, rng: random.Random) -> dict:
    year = pick(rng, YEARS)
    method = "THPT"
    variant = record_id % 8
    if variant in {0, 1, 2}:
        ranking_focus = [
            MAJOR_BY_NAME["Công nghệ thông tin"],
            MAJOR_BY_NAME["Kỹ thuật phần mềm"],
            MAJOR_BY_NAME["Khoa học máy tính"],
            MAJOR_BY_NAME["Trí tuệ nhân tạo"],
            MAJOR_BY_NAME["An toàn thông tin"],
            MAJOR_BY_NAME["Hệ thống thông tin"],
            MAJOR_BY_NAME["Marketing"],
            MAJOR_BY_NAME["Kinh tế quốc tế"],
        ]
        major = ranking_focus[(record_id // 8) % len(ranking_focus)]
        candidates = sorted(schools_for_major(major), key=lambda s: score_for(s, major, year, method), reverse=True)[:8]
        doc = doc_ranking(f"D{record_id}_R", candidates, major, year, method)
        top_n = 5
        top = candidates[:top_n]
        if variant == 0:
            question = f"Xếp hạng 5 trường có điểm chuẩn khối A00 ngành {major.name} cao nhất năm {year} kèm học phí tương ứng từng trường"
        elif variant == 1:
            question = f"Danh sách top 5 trường điểm chuẩn ngành {major.name} năm {year} cao nhất và học phí dự kiến?"
        else:
            question = f"Trường nào có điểm chuẩn ngành {major.name} cao nhất năm {year}, so sánh kèm học phí?"
        rows = [
            f"| {idx} | {s.name} | {score_for(s, major, year, method)} | {format_money(tuition_for(s, major, year))} |"
            for idx, s in enumerate(top, 1)
        ]
        bullets = [
            f"Top {top_n} được sắp xếp giảm dần theo **điểm chuẩn**, không dùng điểm sàn.",
            f"Trường dẫn đầu là **{top[0].name}** với **{score_for(top[0], major, year, method)}** điểm.",
            "Học phí lấy từ cùng bảng tổng hợp trong tài liệu.",
        ]
        table = "| Hạng | Trường | Điểm chuẩn | Học phí |\n|---:|---|---:|---:|\n" + "\n".join(rows)
        answer = answer_header(f"Xếp hạng điểm chuẩn {major.name} {year}", bullets, table, "Chỉ xếp hạng các trường có đủ cả điểm chuẩn và học phí trong tài liệu.", [doc])
        docs = [doc]
        task_type = "ranking"
    elif variant in {3, 4}:
        major = pick(rng, [m for m in MAJORS if m.group in {"Công nghệ", "Kinh tế", "Ngôn ngữ", "Kỹ thuật"} and len(schools_for_major(m)) >= 2])
        s1, s2 = rng.sample(schools_for_major(major), 2)
        docs = [
            doc_admission(f"D{record_id}_A1", s1, year, relevant_majors(record_id, major, school=s1), method),
            doc_admission(f"D{record_id}_A2", s2, year, relevant_majors(record_id + 1, major, school=s2), method),
            doc_tuition(f"D{record_id}_T1", s1, year, [major]),
            doc_tuition(f"D{record_id}_T2", s2, year, [major]),
        ]
        question = f"So sánh điểm chuẩn và học phí ngành {major.name} của {s1.acronym} và {s2.acronym} năm {year}?"
        score1, score2 = score_for(s1, major, year), score_for(s2, major, year)
        fee1, fee2 = tuition_for(s1, major, year), tuition_for(s2, major, year)
        better = s1.acronym if score1 >= score2 else s2.acronym
        cheaper = s1.acronym if fee1 <= fee2 else s2.acronym
        bullets = [f"{s1.acronym}: **{score1}** điểm, học phí **{format_money(fee1)}**.", f"{s2.acronym}: **{score2}** điểm, học phí **{format_money(fee2)}**.", f"Điểm cao hơn: **{better}**; học phí thấp hơn: **{cheaper}**."]
        table = f"| Trường | Điểm chuẩn | Học phí |\n|---|---:|---:|\n| {s1.acronym} | {score1} | {format_money(fee1)} |\n| {s2.acronym} | {score2} | {format_money(fee2)} |"
        answer = answer_header(f"So sánh {major.name} {year}", bullets, table, "So sánh chỉ áp dụng cho cùng ngành và cùng phương thức THPT.", docs)
        task_type = "comparison"
    else:
        filter_focus = [
            MAJOR_BY_NAME["Công nghệ thông tin"],
            MAJOR_BY_NAME["Kỹ thuật phần mềm"],
            MAJOR_BY_NAME["An toàn thông tin"],
            MAJOR_BY_NAME["Kế toán"],
            MAJOR_BY_NAME["Logistics và Quản lý chuỗi cung ứng"],
            MAJOR_BY_NAME["Kỹ thuật điều khiển và tự động hóa"],
        ]
        major = filter_focus[(record_id // 8) % len(filter_focus)]
        city = pick(rng, ["Hà Nội", "TP.HCM"])
        min_score = round(24 + (stable_int(record_id, "min") % 40) / 10, 1)
        max_fee = pick(rng, [35, 40, 45, 50, 60])
        schools = schools_for_major(major, city)
        doc = doc_ranking(f"D{record_id}_R", schools, major, year, method)
        valid = [s for s in schools if score_for(s, major, year) <= min_score and tuition_for(s, major, year) <= max_fee]
        valid = sorted(valid, key=lambda s: (score_for(s, major, year), tuition_for(s, major, year)), reverse=True)[:5]
        question = f"Với {min_score} điểm A00, muốn học ngành {major.name} ở {city} học phí dưới {max_fee} triệu thì nên chọn trường nào?"
        if valid:
            rows = [f"| {s.name} | {score_for(s, major, year)} | {format_money(tuition_for(s, major, year))} | Phù hợp |" for s in valid]
            bullets = [f"Có **{len(valid)}** lựa chọn thỏa điều kiện điểm chuẩn không vượt {min_score} và học phí dưới {max_fee} triệu.", f"Ưu tiên các trường đúng thành phố **{city}** và đúng ngành **{major.name}**.", "Không đưa vào các trường thiếu học phí hoặc vượt ngưỡng."]
            table = "| Trường | Điểm chuẩn | Học phí | Đánh giá |\n|---|---:|---:|---|\n" + "\n".join(rows)
        else:
            bullets = [f"Không có trường nào trong tài liệu thỏa đồng thời điểm chuẩn không vượt {min_score} và học phí dưới {max_fee} triệu.", "Có thể nới điều kiện học phí, mở rộng địa bàn hoặc xét ngành gần."]
            table = None
        answer = answer_header(f"Tư vấn chọn trường {major.name} {year}", bullets, table, "Bộ lọc phải thỏa đồng thời ngành, địa điểm, điểm chuẩn và học phí.", [doc])
        docs = [doc]
        task_type = "filter_recommendation"
    return build_record(record_id, "so_sanh_tu_van_xep_hang", task_type, question, maybe_add_distractor(docs, rng, record_id), answer, {"major": major.name, "year": year})


def gen_no_answer(record_id: int, rng: random.Random) -> dict:
    school = pick(rng, SCHOOLS)
    major = pick(rng, offered_majors(school))
    year = pick(rng, YEARS)
    docs = [doc_floor_only(f"D{record_id}_F", school, year, relevant_majors(record_id, major, school=school))]
    variant = record_id % 5
    if variant == 0:
        question = f"Điểm chuẩn ngành {major.name} trường {school.acronym} năm {year} là bao nhiêu?"
        reason = "Tài liệu chỉ có điểm sàn/điểm nhận hồ sơ, không có điểm chuẩn trúng tuyển."
    elif variant == 1:
        question = f"Học phí ngành {major.name} trường {school.acronym} năm {year} là bao nhiêu?"
        reason = "Tài liệu không có bảng học phí."
    elif variant == 2:
        question = f"Xếp hạng 5 trường có điểm chuẩn ngành {major.name} cao nhất năm {year} kèm học phí?"
        reason = "Tài liệu chỉ nói về một trường và không có học phí, nên không đủ để xếp hạng top 5."
    elif variant == 3:
        question = f"Tỉ lệ việc làm sau tốt nghiệp ngành {major.name} của {school.acronym} là bao nhiêu?"
        reason = "Tài liệu không chứa khảo sát việc làm sau tốt nghiệp."
    else:
        question = f"Giấy chứng nhận kiểm định ngành {major.name} của {school.acronym} còn hiệu lực đến năm nào?"
        reason = "Tài liệu không cung cấp thông tin kiểm định."
    answer = answer_header(
        "Không đủ dữ liệu để trả lời",
        [reason, "Không nên suy luận từ điểm sàn sang điểm chuẩn hoặc từ tài liệu không đúng loại thông tin.", "Cần bổ sung đúng thông báo điểm chuẩn/học phí/kiểm định tương ứng."],
        None,
        "Trả lời thiếu dữ liệu là bắt buộc khi tài liệu không chứa trường thông tin được hỏi.",
        docs,
    )
    return build_record(record_id, "khong_du_du_lieu_guardrail", "no_answer", question, maybe_add_distractor(docs, rng, record_id), answer, {"school": school.acronym, "major": major.name, "year": year})


def maybe_add_distractor(docs: list[dict], rng: random.Random, record_id: int) -> list[dict]:
    noisy_docs = docs + [distractor_doc(f"D{record_id}_N1", rng)]
    if record_id % 5 == 0:
        noisy_docs.append(distractor_doc(f"D{record_id}_N2", rng))
    return noisy_docs


GENERATOR_COUNTS: list[tuple[str, int, Callable[[int, random.Random], dict]]] = [
    ("tuyen_sinh_diem", 750, gen_score),
    ("tuyen_sinh_chi_tieu_phuong_thuc", 450, gen_quota_method),
    ("hoc_phi_tai_chinh", 600, gen_tuition),
    ("hoc_vu_sinh_vien", 550, gen_academic),
    ("cong_khai_kiem_dinh", 800, gen_public),
    ("nganh_dao_tao_nghe_nghiep", 650, gen_program),
    ("so_sanh_tu_van_xep_hang", 950, gen_comparison),
    ("khong_du_du_lieu_guardrail", 250, gen_no_answer),
]


def read_question_templates(root: Path) -> dict[str, list[str]]:
    files = [root / "danh_sach_cau_hoi_finetune.md", root / "thong_tin_cong_khai_kiem_dinh.md"]
    result: dict[str, list[str]] = {}
    for path in files:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        result[path.name] = [line.strip()[1:].strip() for line in text.splitlines() if line.strip().startswith("-") and "?" in line]
    return result


def validate_records(records: list[dict]) -> None:
    if len(records) != 5000:
        raise ValueError(f"Expected 5000 records, got {len(records)}")
    questions = [r["question"] for r in records]
    duplicates = [q for q, count in Counter(questions).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate questions found: {duplicates[:3]}")
    for record in records:
        if "[Tên" in record["question"] or "[Năm" in record["question"]:
            raise ValueError(f"Unreplaced placeholder in {record['id']}: {record['question']}")
        if not record["documents"] or not record["answer"].strip():
            raise ValueError(f"Missing document/answer in {record['id']}")
        if len(record["messages"]) != 3:
            raise ValueError(f"Invalid messages in {record['id']}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a synthetic grounded dataset for LoRA + RAFT.")
    parser.add_argument("--out-dir", default="finetune", help="Output directory")
    parser.add_argument("--seed", type=int, default=20260521)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    templates = read_question_templates(root)
    records: list[dict] = []
    seen_questions: set[str] = set()
    record_id = 1
    for _, count, generator in GENERATOR_COUNTS:
        for _ in range(count):
            # Regenerate on rare duplicate caused by random choices.
            for _attempt in range(100):
                record = generator(record_id, rng)
                if record["question"] not in seen_questions:
                    break
                record_id += 1
            if record["question"] in seen_questions:
                record["question"] = record["question"].rstrip("?") + f" theo tài liệu mã {record_id:05d}?"
            seq_id = len(records) + 1
            record["id"] = f"lora_raft_{seq_id:05d}"
            record["split"] = "train" if seq_id <= 4500 else ("validation" if seq_id <= 4750 else "test")
            record["metadata"]["source_files"] = list(templates.keys())
            record["metadata"]["synthetic"] = True
            records.append(record)
            seen_questions.add(record["question"])
            record_id += 1

    validate_records(records)

    combined_path = out_dir / "lora_raft_dataset_5000.jsonl"
    lora_path = out_dir / "lora_messages_5000.jsonl"
    raft_path = out_dir / "raft_qa_documents_5000.jsonl"
    summary_path = out_dir / "lora_raft_dataset_5000_summary.json"

    write_jsonl(combined_path, records)
    write_jsonl(lora_path, [{"id": r["id"], "split": r["split"], "messages": r["messages"], "metadata": r["metadata"]} for r in records])
    write_jsonl(
        raft_path,
        [
            {
                "id": r["id"],
                "split": r["split"],
                "category": r["category"],
                "task_type": r["task_type"],
                "question": r["question"],
                "answer": r["answer"],
                "documents": r["documents"],
                "oracle_document_ids": [d["doc_id"] for d in r["documents"] if d.get("is_relevant")],
                "distractor_document_ids": [d["doc_id"] for d in r["documents"] if not d.get("is_relevant")],
                "metadata": r["metadata"],
            }
            for r in records
        ],
    )

    summary = {
        "total_records": len(records),
        "seed": args.seed,
        "outputs": {
            "combined": str(combined_path.relative_to(root)),
            "lora_messages": str(lora_path.relative_to(root)),
            "raft_qa_documents": str(raft_path.relative_to(root)),
        },
        "splits": dict(Counter(r["split"] for r in records)),
        "categories": dict(Counter(r["category"] for r in records)),
        "task_types": dict(Counter(r["task_type"] for r in records)),
        "source_template_counts": {name: len(items) for name, items in templates.items()},
        "schema": {
            "combined": ["id", "split", "category", "task_type", "question", "answer", "documents", "messages", "metadata"],
            "document": ["doc_id", "title", "source", "content", "is_relevant"],
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
