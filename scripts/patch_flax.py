import sys
import os
import importlib.util
import types

def apply_patch():
    try:
        spec = importlib.util.find_spec('flax')
        if spec is None:
            print("Flax not found, skipping patch.")
            return
        
        flax_path = os.path.dirname(spec.origin)
        kw_only_path = os.path.join(flax_path, 'linen', 'kw_only_dataclasses.py')
        
        if not os.path.exists(kw_only_path):
            print(f"Could not find {kw_only_path}, skipping patch.")
            return

        with open(kw_only_path, 'r', encoding='utf-8') as f:
            source = f.read()

        # Replacement 1: Remove import flax
        if "import flax\n" in source:
            source = source.replace("import flax\n", "# import flax\n")
            print("Removed 'import flax' to avoid circular dependency.")

        # Replacement 2: Inject inspect.get_annotations logic
        target_block = "  if '__annotations__' not in cls.__dict__:\n    cls.__annotations__ = {}"
        replacement_block = """  if '__annotations__' not in cls.__dict__:
    # Use global inspect
    try:
      cls.__annotations__ = inspect.get_annotations(cls)
    except Exception:
      cls.__annotations__ = {}"""
        
        if target_block in source:
            source = source.replace(target_block, replacement_block)
            print("Injected inspect.get_annotations logic.")
        else:
            print("Target block for annotations fix not found.")

        # Replacement 3: Safe access to __annotations__ later in the file
        old_code = "cls_annotations = cls.__dict__['__annotations__']"
        new_code = "cls_annotations = cls.__dict__.get('__annotations__', {})"
        
        if old_code in source:
            source = source.replace(old_code, new_code)
            print("Patched unsafe __annotations__ access.")
        
        # Create module
        module_name = 'flax.linen.kw_only_dataclasses'
        module = types.ModuleType(module_name)
        
        module.__file__ = kw_only_path
        module.__package__ = 'flax.linen'
        
        # Execute source
        exec(source, module.__dict__)
        
        # Inject into sys.modules
        sys.modules[module_name] = module
        print(f"Injected patched {module_name} into sys.modules")

    except Exception as e:
        print(f"Failed to apply patch: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    apply_patch()
