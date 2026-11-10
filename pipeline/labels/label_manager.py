COMPONENTS = ("nozzle_health", "feeder_health", "motion_system_health")


def prompt_labels(read=input):
    while True:
        state = read("Print condition (1 healthy, 2 failing): ").strip().lower()
        if state not in ("1", "2", "healthy", "failing"):
            continue
        selected = set()
        if state in ("2", "failing"):
            selected = set(read("Affected components (1 nozzle, 2 feeder, 3 motion system; comma-separated): ").replace(",", " ").split())
            if not selected or not selected <= {"1", "2", "3"}:
                print("Select at least one listed component.")
                continue
        labels = {key: "failing" if str(i) in selected else "healthy"
                  for i, key in enumerate(COMPONENTS, 1)}
        print(labels)
        if read("Confirm labels and start recording? [y/N] ").strip().lower() in ("y", "yes"):
            return labels
