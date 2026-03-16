from ultralytics import YOLO
import torch




class PlayerTracker:
    """
    Player tracker using YOLO detection model
    """

    def __init__(self, model_path, conf):
        self.model = YOLO(model_path)
        self.conf = conf
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        #getting hoop class to filter out of drawing it
        #thats it nothign else 
        player_name_to_id = {v: k for k, v in self.model.names.items()}
        self.HOOP_CLASS_ID = player_name_to_id.get('Hoop')

    
    def get_player_tracks(self, frames):
        player_tracks = []
        for frame in frames:
            player_results = self.model.track(
                frame, conf = self.conf, tracker = "botsort.yaml",
                persist = True, verbose = False, agnostic_nms = True, device=self.device)
            # Filter out hoops before storing
            if player_results[0].boxes.id is not None:
                filtered = [
                    (box, tid, cls)
                    for box, tid, cls in zip(
                        player_results[0].boxes.xyxy,
                        player_results[0].boxes.id,
                        player_results[0].boxes.cls
                    )
                    if self.HOOP_CLASS_ID is None or int(cls) != self.HOOP_CLASS_ID
                ]
            else:
                filtered = []
            
            player_tracks.append(filtered)
        
        return player_tracks