from tools.research.run_multiview_semantic_oracle_v4_r2 import normalize_local_image_sources_v1


def test_file_uri_is_normalized_to_local_path_without_mutating_input():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "V0"},
                {
                    "type": "image",
                    "image": "file:///content/realsas_semantic_oracle_v4/output/v4_work/source_views/V0.png",
                    "resized_height": 512,
                    "resized_width": 512,
                },
            ],
        }
    ]
    out = normalize_local_image_sources_v1(messages)
    assert out[0]["content"][1]["image"] == "/content/realsas_semantic_oracle_v4/output/v4_work/source_views/V0.png"
    assert messages[0]["content"][1]["image"].startswith("file://")


def test_http_and_plain_local_paths_are_left_unchanged():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "https://example.com/a.png"},
                {"type": "image", "image": "/content/a.png"},
            ],
        }
    ]
    out = normalize_local_image_sources_v1(messages)
    assert out == messages
