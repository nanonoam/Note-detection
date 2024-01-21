import math 
def get_y(m, k, delta_x, angle, x):
   
#    מחשב לכל מרחק מהפיו פיו גובה
#    m: מסה (kg)
#    k: קבוע הקפיץ (N/m)
#    delta_x: כמה הקפיץ היתקבץ (m) 
#    angle: זווית הפיו פיו (degrees)
#    x: מרחק מהמטרה (m)
   
   
   # מישובים
   spring_potential_energy = 0.5 * k * delta_x**2 # אנרגיה פוטנציאלית של הקפיץ
   # v = math.sqrt(2 * spring_potential_energy / m) # מעבירים אנרגיה פותנציאלית לאנרגיה קינתית
   v = 5
   print (v)
   angle_rad = math.radians(angle) # מעבירים את הזווית לרדיאנים
   # מחשב את ה x ו-y של המהירות
   vx = v*math.cos(angle_rad) # x של המהירות
   vy = v*math.sin(angle_rad) # y של המהירוץ
   
   g = 9.81 # תאוצה של קוח הקבידה
   
   # נוסחאות לחישוב טרגקטורי
   T = 2 * v / g # זמן שהאוביקט באוויר
   t = min(T, x/vx) # זמן עד שמגיעים למטרה או לריצפה(מה שמגיע קודם)
   y = vy*t - 0.5*g*t**2 #מחשב את הגובה של האוביקט
   
   return y

# פרמטריפ
mass = 0.01 
k_spring = 46.91311692
compression = 0.073
angle = 30
x_pos = 2.5

# גובה של פיו פיו
height = 1.16

# מוסיף גובה
y = get_y(mass, k_spring, compression, angle, x_pos) + height

print(f"The y hight at x={x_pos} m is: {y} m")