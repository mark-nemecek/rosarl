from typing import Any


def extract_data_from_info(info: dict[str, Any], key: str):
    data = None
    if key in info:
        data = info[key]
    elif "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict) and key in final_info:
            data = final_info[key]
        elif key in info["final_info"][0]:
            data = tuple(i[key] for i in final_info)

    return data
