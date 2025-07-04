import trimesh

def main():
    trimesh.primitives.Box(
        [1.5, 1.0, 0.5]
    ).apply_translation(
        [0., +0.5, -0.25]
    ).export("assets/misc/table.obj")

if __name__ == "__main__":
    main()