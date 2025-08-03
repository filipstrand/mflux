class TurboUtil:
    # Turbo mode configuration
    TURBO_LORA_REPO_ID = "alimama-creative/FLUX.1-Turbo-Alpha"
    TURBO_LORA_FILENAME = "diffusion_pytorch_model.safetensors"
    TURBO_GUIDANCE = 3.5

    @staticmethod
    def apply_turbo_settings(args):
        if not args.turbo:
            return

        # Add turbo LoRA parameters
        # Safely initialize lora_names, preserving any existing values
        if not hasattr(args, "lora_names") or args.lora_names is None:
            args.lora_names = []

        # Convert CLI's singular lora_name to plural lora_names list if provided
        if hasattr(args, "lora_name") and args.lora_name:
            # Ensure lora_names is a proper list (handle Mock objects in tests)
            if not isinstance(args.lora_names, list):
                args.lora_names = []
            if args.lora_name not in args.lora_names:
                args.lora_names.append(args.lora_name)

        # Handle LoRA scales when turbo adds an additional LoRA
        # If user provided lora_paths but no lora_scales, we need to add default scales
        if hasattr(args, "lora_paths") and args.lora_paths and isinstance(args.lora_paths, list):
            if not hasattr(args, "lora_scales") or args.lora_scales is None:
                # Default to 1.0 for all existing user LoRAs
                args.lora_scales = [1.0] * len(args.lora_paths)

        # Set repo ID for Hugging Face download
        args.lora_repo_id = TurboUtil.TURBO_LORA_REPO_ID

        # Add the turbo LoRA name (exact filename from the repository)
        if isinstance(args.lora_names, list) and TurboUtil.TURBO_LORA_FILENAME not in args.lora_names:
            args.lora_names.append(TurboUtil.TURBO_LORA_FILENAME)

        # Only set guidance default if not explicitly set by user
        # Note: Steps are left to user control
        if args.guidance is None:
            args.guidance = TurboUtil.TURBO_GUIDANCE
