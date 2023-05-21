import streamlit as st


title = 'Swan Classification'
st.set_page_config(
    page_title=title,
    page_icon="👋",
)


with st.spinner('Инициализация нейронной сети...'):
    import numpy as np
    from ultralytics import YOLO
    import torch
    # from transformers import OwlViTProcessor, OwlViTForObjectDetection
    from PIL import Image
    import cv2
    import gdown
    
    gdown.download("https://drive.google.com/file/d/1ifKeWqsp16X4S14JYjKB3tPELviBq3Zv/view?usp=share_link", "swan_best_1.pt", quiet=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO('./swan_best_1.pt')
    yolo.to(device)

    ####

    # classifier = timm.create_model('coatnet_rmlp_1_rw2_224.sw_in12k', pretrained=False, drop_rate=0.3, num_classes=4) # , global_pool=''
    # checkpoint = torch.load("./data/coatnet_rmlp_1_rw2_224.sw_in12k_33_bs32_mup0.0_met0.980.ckpt" , map_location='cpu')

    # classifier.load_state_dict(checkpoint['model'])
    # classifier = classifier.to(device)
    # classifier.eval()
    ####################

label2cls = ["Шипун", "Малый", "Кликун"]

def recognise(img, sensitivity):
    sensitivity = sensitivity / 5 * 3 + 1
    threshold = 1 / sensitivity
    H, W, _ = img.shape
    bgr_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pred = yolo(bgr_img, verbose=False)[0]
    boxes = pred.boxes.xyxy.tolist()
    labels = pred.boxes.cls.tolist()
    scores = pred.boxes.conf.tolist()


    # x = val_transform(image=img)['image']
    # x = x.unsqueeze(0).to(device)
    # x = classifier(x)[0].detach().cpu().numpy()
    # x = x.argmax().item()
    # x = label2cls[x]

    draw_img = img.copy()
    counter = {
        'Шипун': 0,
        'Кликун': 0,
        'Малый': 0
    }

    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        
        score = round(score, 2)
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        # crop = img[y1:y2, x1:x2].copy()
        thickness = (H + W) // 400
        cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), thickness)

        label = int(label)
        cls = label2cls[label]

        # crop = val_transform(image=crop)['image']
        # crop = crop.unsqueeze(0).to(device)
        # prob = classifier(crop)[0].detach().cpu().numpy()
        # label = prob.argmax().item()
        # cls = label2cls[label]

        counter[cls] += 1
        print(cls)
        text = {'Шипун': 'Mute', 'Кликун': 'Whooper', 'Малый': 'Tundra'}[cls]
        cv2.putText(draw_img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 100), thickness, cv2.LINE_AA)

    x = max(counter.keys(), key=lambda x: counter[x])
    return draw_img, x, counter['Шипун'], counter['Кликун'], counter['Малый']

#################

st.markdown(f"# {title} :sunglasses:")
st.success("## Команда: ML Rocks")
st.markdown("#")

st.divider()

st.markdown("### Обнаруживаем лебедей")

img_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
sensitivity  = st.slider("Чувствительность", min_value=0.0, max_value=5.0, value=0.37, step=0.01)

if img_file is not None:
    img = Image.open(img_file).convert('RGB') 
    img = np.array(img)

    with st.spinner('Обработка изображения'):
        img, cls, mute_n, whooper_n, tundra_n = recognise(img, sensitivity=sensitivity)
    
    st.image(img, caption=f"Результат распознавания")
    st.write(f"Общая классификация:", cls)
    st.write(f"Количество лебедей-шипунов:", mute_n)
    st.write(f"Количество дебедей-кликунов:", whooper_n)
    st.write(f"Количество малых лебедей:", tundra_n)
    st.markdown(f"*Примечание*: Mute - это Шипун, Whooper - Кликун, Tundra - Малый (проблема со шрифтом русского текста)")
    st.markdown(f"**По каким либо вопросам: https://t.me/dl_hello")


st.divider()
