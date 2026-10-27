import re
import cv2
import numpy as np

def Stringparsing(s):
    """（您已实现的函数，此处略）"""
    pattern = r'^(\d+)-(\d+_\d+)-(\d+&\d+_\d+&\d+)-((?:\d+&\d+_){3}\d+&\d+)-(\d+_\d+_\d+_\d+_\d+_\d+_\d+)-(\d+)-(\d+)\.jpg$'
    match = re.match(pattern, s)
    if not match:
        raise ValueError(f"文件名格式不匹配: {s}")
    groups = match.groups()
    return {
        'area': groups[0],
        'tilt': groups[1],
        'bbox': groups[2],
        'vertices': groups[3],
        'plate': groups[4],
        'brightness': groups[5],
        'blurriness': groups[6]
    }

def parse_coord_str(coord_str):
    """将 'x&y' 转为 (x, y) 整数元组"""
    x, y = coord_str.split('&')
    return (int(x), int(y))

def annotate_image(image_path, filename, output_path=None, show=True):
    """
    在图像上标注车牌区域。

    参数:
        image_path : str   - 原始图像文件路径（例如 './02.jpg'）
        filename    : str   - 对应的 CCPD 文件名（用于解析坐标）
        output_path : str   - 保存标注图像的路径，默认为原路径加 '_annotated'
        show        : bool  - 是否显示图像窗口
    """
    # 1. 解析文件名获取坐标
    data = Stringparsing(filename)
    bbox_str = data['bbox']          # "x1&y1_x2&y2"
    vertices_str = data['vertices']  # "x1&y1_x2&y2_x3&y3_x4&y4"

    # 转换为整数坐标元组
    bbox_parts = bbox_str.split('_')
    x1, y1 = parse_coord_str(bbox_parts[0])
    x2, y2 = parse_coord_str(bbox_parts[1])

    vertex_parts = vertices_str.split('_')
    # 注意：顶点顺序为 右下 -> 左下 -> 左上 -> 右上
    pts = [parse_coord_str(p) for p in vertex_parts]  # 列表 [(x,y), ...]

    # 2. 读取图像
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")

    # 3. 绘制边界框（绿色矩形，线宽2）
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 4. 绘制四个顶点（红色圆点，半径6）
    for (x, y) in pts:
        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)  # -1 填充圆

    # 5. 连接四个顶点形成多边形（蓝色线条，线宽2）
    #   注意 polylines 需要点集为 numpy 数组，形状 (N,1,2)
    pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_array], isClosed=True, color=(255, 0, 0), thickness=2)

    # 6. （可选）在左上角显示车牌号
    plate_code = data['plate']  # 例如 "0_0_20_24_21_30_27"
    cv2.putText(img, f"Plate: {plate_code}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 7. 保存结果
    if output_path is None:
        # 在原始路径基础上加后缀
        if '.' in image_path:
            base, ext = image_path.rsplit('.', 1)
            output_path = f"{base}_annotated.{ext}"
        else:
            output_path = image_path + "_annotated.jpg"
    cv2.imwrite(output_path, img)
    print(f"标注图像已保存至: {output_path}")

    # 8. 显示图像（按任意键关闭窗口）
    if show:
        cv2.imshow("Annotated Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img  # 可选返回图像数组


# ---------- 测试 ----------
if __name__ == "__main__":
    # 示例文件名（与您的一致）
    filename = "02-1_4-352&371_544&458-539&454_352&458_357&375_544&371-0_0_20_24_21_30_27-19-2.jpg"
    # 假设原始图像名为 "02.jpg"（实际请根据您的图像路径调整）
    image_path = "02.jpg"  # 请确保此文件存在

    try:
        annotate_image(image_path, filename, show=True)
    except FileNotFoundError as e:
        print(e)
        print("请将图像文件放在当前目录，并修改 image_path 变量。") 