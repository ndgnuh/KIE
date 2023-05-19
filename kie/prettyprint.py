def get_fullstring(start_point: int, links: list):
    sequence = [start_point]  # Khởi tạo chuỗi với số bắt đầu

    current_number = start_point  # Số hiện tại trong chuỗi
    found_match = True
    while found_match:
        found_match = False
        # Tìm số sau của cặp [số trước, số sau] trong input
        for pair in links:
            if pair[0] == current_number:
                next_number = pair[1]
                found_match = True
                sequence.append(next_number)  # Thêm số sau vào chuỗi
                current_number = next_number  # Cập nhật số hiện tại
                break

    return sequence


def simple_postprocess(sample, fields: list):
    #     links: box to box
    #     classes: class to box
    #     texts: texts list
    #     fields: classes list
    links = sample.links
    classes = sample.classes
    texts = sample.texts
    label_text_list = []

    for key in classes:
        class_name = fields[classes[key]]
        start_point = int(key)
        ids_sequence = get_fullstring(start_point, links)
        string_texts = [texts[x] for x in ids_sequence]
        label_text_list.append({class_name: " ".join(string_texts)})
    return label_text_list
