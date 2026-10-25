import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_320_fpn

from logoloc.eval.predictors import detect_with_proposals


def test_detect_with_proposals_matches_model_output():
    torch.manual_seed(0)
    model = fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, weights_backbone=None, num_classes=3).eval()
    model.roi_heads.score_thresh = 0.0
    image = torch.rand(3, 240, 320)

    with torch.no_grad():
        reference = model([image])[0]
        output = detect_with_proposals(model, image)

    assert len(output["boxes"]) > 0
    assert torch.allclose(output["boxes"], reference["boxes"])
    assert torch.equal(output["labels"], reference["labels"])
    assert torch.allclose(output["scores"], reference["scores"])
    assert output["proposals"].shape == output["boxes"].shape
