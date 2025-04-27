import cv2
import numpy as np
import os
from glob import glob
from tqdm import tqdm


class OrthophotoGenerator:
    def __init__(self):
     
        self.sift = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    def load_images(self, folder_path):
        """Загрузка и сортировка изображений из папки"""
        img_paths = sorted(glob(os.path.join(folder_path, "*.jpg"))) + \
                    sorted(glob(os.path.join(folder_path, "*.jpeg"))) + \
                    sorted(glob(os.path.join(folder_path, "*.png")))

        print(f"Найдено {len(img_paths)} изображений")
        images = []
        for path in tqdm(img_paths, desc="Загрузка изображений"):
            img = cv2.imread(path)
            if img is not None:

                h, w = img.shape[:2]
                if w > 2000 or h > 2000:
                    img = cv2.resize(img, (int(w * 0.5), int(h * 0.5)),
                                     interpolation=cv2.INTER_AREA)
                images.append(img)
        return images

    def match_features(self, img1, img2):
        """Сопоставление особенностей между двумя изображениями"""
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.sift.detectAndCompute(gray1, None)
        kp2, des2 = self.sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            return None, None

        matches = self.matcher.knnMatch(des1, des2, k=2)


        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        if len(good) < 10:
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        return src_pts, dst_pts

    def create_orthophoto(self, images):
        """Создание ортофотоплана из набора изображений"""
        if len(images) < 2:
            raise ValueError("Нужно минимум 2 изображения")

        print("Начинаем процесс сшивания...")


        status, result = self.stitcher.stitch(images)

        if status == cv2.Stitcher_OK:
            return result
        else:
            print(f"Ошибка сшивания (код {status}), пробуем ручной метод...")
            return self.manual_stitching(images)

    def manual_stitching(self, images):
        """Ручное сшивание, если автоматическое не сработало"""
        base_img = images[0]

        for i in tqdm(range(1, len(images)), desc="Ручное сшивание"):
            src_pts, dst_pts = self.match_features(images[i], base_img)

            if src_pts is None:
                print(f"Не удалось сопоставить изображение {i + 1}")
                continue

            H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if H is None:
                print(f"Не удалось найти гомографию для изображения {i + 1}")
                continue


            h, w = base_img.shape[:2]
            warped = cv2.warpPerspective(images[i], H, (w * 2, h * 2))


            mask = np.zeros((h * 2, w * 2), dtype=np.uint8)
            cv2.fillConvexPoly(mask, np.int32(dst_pts * 2), 255)

            base_img = cv2.seamlessClone(
                warped,
                cv2.resize(base_img, (w * 2, h * 2)),
                mask,
                (w, h),
                cv2.NORMAL_CLONE
            )


            gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x, y, w, h = cv2.boundingRect(contours[0])
            base_img = base_img[y:y + h, x:x + w]

        return base_img


if __name__ == "__main__":
  
    input_folder = ".\\4thAve"
    output_path = ".\\4thAve\\orthophoto_result\\orthophoto_result.jpg"


    cv2.ocl.setUseOpenCL(False)


    generator = OrthophotoGenerator()


    images = generator.load_images(input_folder)

    if len(images) < 2:
        print("Недостаточно изображений для обработки")
    else:

        result = generator.create_orthophoto(images)


        cv2.imwrite(output_path, result)
        print(f"Ортофотоплан сохранен в {output_path}")


        cv2.imshow("Orthophoto", cv2.resize(result, (1000, 600)))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
