class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, target_ap=0.97):
        self.patience = patience
        self.min_delta = min_delta
        self.target_ap = target_ap
        self.best_loss = None
        self.wait = 0

    def check(self, val_loss, val_ap):
        if val_ap >= self.target_ap:
            if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping triggered: AP {val_ap:.3f} > {self.target_ap} and no val loss improvement for {self.patience} runs.")
                return True
        return False
