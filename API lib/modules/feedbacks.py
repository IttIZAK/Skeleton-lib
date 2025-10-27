# modules/feedbacks.py

def feedback_squat(conf, hold=0.0):
    """Feedback สำหรับ Squat แบบละเอียด"""
    if conf > 0.90:
        return "เพอร์เฟ็กต์! ย่อลึกและหลังตรง ๆ"
    elif conf > 0.75:
        return "ดีมาก! รักษาท่านี้ไว้"
    elif conf > 0.60:
        return "ดีแล้ว แต่ลองย่อลึกกว่านี้อีกนิด"
    elif conf > 0.40:
        return "ย่อเข่าให้ต่ำลงจนสะโพกต่ำกว่าเข่า"
    elif conf > 0.20:
        return "ลองยืนตรง แล้วค่อย ๆ ย่อลงช้า ๆ"
    else:
        return "เข้าท่ายืนตรงก่อน แล้วเริ่มย่อลง"

def feedback_pushup(conf, hold=0.0):
    """Feedback สำหรับ Push-ups"""
    if conf > 0.90:
        return "สุดยอด! ท่าสมบูรณ์แบบ"
    elif conf > 0.75:
        return "ดีเลิศ! รักษาลำตัวให้ตรง"
    elif conf > 0.60:
        return "ดีแล้ว ลองงอศอกลึกกว่านี้อีกหน่อย"
    elif conf > 0.40:
        return "งอศอกให้มากขึ้น หน้าอกใกล้พื้น"
    elif conf > 0.20:
        return "รักษาลำตัวให้ตรงตลอด และงอศอกช้า ๆ"
    else:
        return "เข้าท่าอัพก่อน แขนยืดตรง"

def feedback_plank(conf, hold=0.0):
    """Feedback สำหรับ Plank พร้อมเวลา"""
    if conf > 0.85:
        return f"เยี่ยมมาก! ค้างไว้ได้ {hold:.1f} วินาที"
    elif conf > 0.70:
        return f"ดี! ({hold:.1f}s) แต่ลองยกสะโพกเล็กน้อย"
    elif conf > 0.50:
        return "ปรับลำตัวให้ตรงเป็นแนวเดียวกัน"
    elif conf > 0.30:
        return "ยกสะโพกขึ้นให้สูงกว่านี้"
    else:
        return "เข้าท่าพลังก์ ศอกแนบพื้น ลำตัวตรง"

def feedback_situp(conf, hold=0.0):
    """Feedback สำหรับ Sit-ups"""
    if conf > 0.85:
        return "เพอร์เฟ็กต์! นั่งขึ้นได้เต็มที่"
    elif conf > 0.70:
        return "ดีมาก! ยกตัวขึ้นได้ดี"
    elif conf > 0.50:
        return "ลองงอลำตัวขึ้นอีกหน่อย"
    elif conf > 0.30:
        return "ใช้กล้ามท้องดึงตัวขึ้น ไม่ใช่คอ"
    else:
        return "นอนราบก่อน แล้วค่อยนั่งขึ้นช้า ๆ"

def feedback_lunge(conf, hold=0.0):
    """Feedback สำหรับ Lunge"""
    if conf > 0.85:
        return "สมบูรณ์แบบ! เข่าได้มุม 90° พอดี"
    elif conf > 0.70:
        return "ดีมาก! รักษาความสมดุล"
    elif conf > 0.50:
        return "ลองย่อลงอีกนิด เข่าหน้า 90°"
    elif conf > 0.30:
        return "ก้าวขาให้ยาวขึ้น แล้วย่อลง"
    else:
        return "ยืนตรง แล้วก้าวขาหน้าออกไป"

def feedback_dead_bug(conf, hold=0.0):
    """Feedback สำหรับ Dead Bug"""
    if conf > 0.85:
        return "เยี่ยม! เหยียดแขนขาตรงข้ามกันดี"
    elif conf > 0.70:
        return "ดี! รักษาท่านี้สลับข้าง"
    elif conf > 0.50:
        return "เหยียดแขนขาให้ตรงมากขึ้น"
    elif conf > 0.30:
        return "นอนราบ ยกแขนขาขึ้นตรงข้ามกัน"
    else:
        return "นอนหงายก่อน แล้วยกแขนขาขึ้น"

def feedback_side_plank(conf, hold=0.0):
    """Feedback สำหรับ Side Plank"""
    if conf > 0.85:
        return f"สุดยอด! ท่าสมบูรณ์ {hold:.1f}s"
    elif conf > 0.70:
        return f"ดี! ({hold:.1f}s) ยกสะโพกอีกหน่อย"
    elif conf > 0.50:
        return "รักษาลำตัวให้ตรงเป็นแนวเดียว"
    elif conf > 0.30:
        return "ยกสะโพกขึ้นให้สูงขึ้น"
    else:
        return "นอนตะแคง ศอกแนบพื้น ยกตัวขึ้น"

def feedback_russian_twist(conf, hold=0.0):
    """Feedback สำหรับ Russian Twist"""
    if conf > 0.85:
        return "เยี่ยม! หมุนลำตัวได้ดี"
    elif conf > 0.70:
        return "ดี! รักษาจังหวะการหมุน"
    elif conf > 0.50:
        return "โน้มตัวไปหลังอีกนิด และหมุนให้มากขึ้น"
    elif conf > 0.30:
        return "นั่งโน้มตัวไปหลัง แล้วหมุนซ้าย-ขวา"
    else:
        return "นั่งก่อน โน้มตัวไปหลัง เตรียมหมุน"

def feedback_lying_leg_raises(conf, hold=0.0):
    """Feedback สำหรับ Lying Leg Raises"""
    if conf > 0.85:
        return "เพอร์เฟ็กต์! ยกขาตรงขึ้นได้เต็มที่"
    elif conf > 0.70:
        return "ดีมาก! ยกขาได้ดี"
    elif conf > 0.50:
        return "ยกขาให้สูงขึ้นอีก ตั้งฉากกับพื้น"
    elif conf > 0.30:
        return "รักษาขาให้ตรงและชิดกัน ยกช้า ๆ"
    else:
        return "นอนราบ ขาชิดกัน เตรียมยกขึ้น"

# Dictionary สำหรับเรียกใช้ง่าย
FEEDBACKS = {
    "Bodyweight Squat": feedback_squat,
    "Push-ups": feedback_pushup,
    "Plank": feedback_plank,
    "Sit-ups": feedback_situp,
    "Lunge (Split Squat)": feedback_lunge,
    "Dead Bug": feedback_dead_bug,
    "Side Plank": feedback_side_plank,
    "Russian Twist": feedback_russian_twist,
    "Lying Leg Raises": feedback_lying_leg_raises,
}