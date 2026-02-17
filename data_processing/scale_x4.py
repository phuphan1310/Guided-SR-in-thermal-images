import cv2
import os

# ====== THIẾT LẬP ======
input_dir = r"C:/Users/ADMIN/MySGNet/data/sugarbeet2016/gt"
output_dir = r"C:/Users/ADMIN/MySGNet/data/sugarbeet2016/depth"
scale = 4   # downsample x4
# ========================

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpeg", ".png", ".tiff", ".tif")):

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Đọc ảnh giữ nguyên bit depth
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

        if img is None:
            print(f"❌ Không đọc được: {filename}")
            continue

        h, w = img.shape[:2]

        # ===== Crop về bội số của scale =====
        new_h = (h // scale) * scale
        new_w = (w // scale) * scale

        if new_h != h or new_w != w:
            img = img[:new_h, :new_w]

            # 🔥 Lưu đè lên ảnh gốc
            cv2.imwrite(input_path, img)

            print(f"✂ Cropped & overwritten: {filename} ({h}x{w} → {new_h}x{new_w})")

        # ===== Downsample =====
        lr = cv2.resize(
            img,
            (new_w // scale, new_h // scale),
            interpolation=cv2.INTER_AREA
        )

        cv2.imwrite(output_path, lr)

        print(f"✔ Downsampled: {filename}")

print("=== Hoàn tất ===")
