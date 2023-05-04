def inference(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            probs = model(imgs)

            probs = probs.cpu().detach().numpy()

            preds = probs > 0.5
            preds = preds.astype(int)
            predictions += preds.tolist()
    return predictions
