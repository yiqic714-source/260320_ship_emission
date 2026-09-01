from pathlib import Path

import xarray as xr


DATA_ROOT = Path(
    "/data/chenyiqi/260320_ship_emission/processed_data/ml_grid_data"
)

# 如果你的实际路径还是 /home/chenyiqi/...，改成：
# DATA_ROOT = Path(
#     "/home/chenyiqi/260320_ship_emission/processed_data/ml_grid_data"
# )


def main():
    nc_files = sorted(DATA_ROOT.rglob("*.nc"))

    print(f"共找到 {len(nc_files)} 个 nc 文件")
    print()

    files_with_any_o3 = []
    files_with_o3_1000 = []
    files_without_o3_1000 = []

    all_o3_variables = set()

    for i, path in enumerate(nc_files, 1):
        try:
            with xr.open_dataset(path) as ds:
                variables = list(ds.data_vars)

                # 所有 o3 相关变量
                o3_vars = [
                    var
                    for var in variables
                    if var.lower().startswith("o3")
                ]

                if o3_vars:
                    files_with_any_o3.append(
                        (path, o3_vars)
                    )
                    all_o3_variables.update(o3_vars)

                if "o3_1000" in ds.data_vars:
                    files_with_o3_1000.append(path)
                else:
                    files_without_o3_1000.append(path)

        except Exception as e:
            print(f"[读取失败] {path}")
            print(f"    {e}")

        if i % 100 == 0:
            print(
                f"已检查 {i}/{len(nc_files)} 个文件..."
            )

    print()
    print("=" * 80)
    print("所有发现的 O3 变量")
    print("=" * 80)

    for var in sorted(all_o3_variables):
        print(var)

    print()
    print("=" * 80)
    print(
        f"存在任意 O3 变量的文件："
        f"{len(files_with_any_o3)} 个"
    )
    print("=" * 80)

    for path, o3_vars in files_with_any_o3:
        print(path)
        print(
            "    O3 variables: "
            + ", ".join(o3_vars)
        )

    print()
    print("=" * 80)
    print(
        f"存在 o3_1000 的文件："
        f"{len(files_with_o3_1000)} 个"
    )
    print("=" * 80)

    for path in files_with_o3_1000:
        print(path)

    print()
    print("=" * 80)
    print(
        f"不存在 o3_1000 的文件："
        f"{len(files_without_o3_1000)} 个"
    )
    print("=" * 80)

    for path in files_without_o3_1000:
        print(path)

    print()
    print("=" * 80)
    print("汇总")
    print("=" * 80)
    print(f"NC 文件总数       : {len(nc_files)}")
    print(
        f"包含任意 O3       : "
        f"{len(files_with_any_o3)}"
    )
    print(
        f"包含 o3_1000      : "
        f"{len(files_with_o3_1000)}"
    )
    print(
        f"不包含 o3_1000    : "
        f"{len(files_without_o3_1000)}"
    )


if __name__ == "__main__":
    main()