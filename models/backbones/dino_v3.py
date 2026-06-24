import timm


# Only the feat_res3 block is consumed (feat_res4 is fabricated from RoI-pooled feat_res3 by
# DinoRes4Head), so out_indices is pruned to that single block per variant; timm features_only
# returns a one-element NCHW list and build_backbone reads feats[0]. (Was the 3rd of 4 blocks.)
vit_model_to_out_indices = {
    'vit_base_patch16_dinov3.lvd1689m': (7,),
    'vit_large_patch16_dinov3.lvd1689m': (17,),
    'vit_huge_plus_patch16_dinov3.lvd1689m': (23,),
    'vit_7b_patch16_dinov3.lvd1689m': (29,),
}


def _create_dino_v3(model_name):
    return timm.create_model(
        model_name=model_name,
        features_only=True,
        dynamic_img_size=True,
        out_indices=vit_model_to_out_indices[model_name],
    )


def dino_v3_b():
    return _create_dino_v3('vit_base_patch16_dinov3.lvd1689m')

def dino_v3_l():
    return _create_dino_v3('vit_large_patch16_dinov3.lvd1689m')

def dino_v3_h_plus():
    return _create_dino_v3('vit_huge_plus_patch16_dinov3.lvd1689m')

def dino_v3_7b():
    return _create_dino_v3('vit_7b_patch16_dinov3.lvd1689m')
