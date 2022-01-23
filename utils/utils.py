def save_params(param_path, args):
    print("[INFO] Saving params!")
    with open(param_path, mode='w') as f:
        arguments = vars(args)
        for key, val in arguments.items():
            f.write(f"{key} {val}\n")
    print("[INFO] Params are successfully saved!")
