# start.py
import subprocess
import sys
import safety_net


def main():
    result = subprocess.run([sys.executable, "main.py"] + sys.argv[1:])

    if result.returncode != 0:
        print("\n" + "="*50)
        print("!!! CRASH DETECTED !!!")
        print("Attempting to revert code to the last working state...")
        print("="*50 + "\n")

        try:
            msg = safety_net.revert()
            if msg:
                print(f"Revert successful: {msg}")
                print("Please try running 'python start.py' again.")
            else:
                print(
                    "No backup state found to revert to. Manual intervention required.")
        except Exception as e:
            print(f"Failed to auto-revert: {e}")


if __name__ == "__main__":
    main()
