import os
import subprocess
class VideoRecorder:
    def __init__(self,
        env,
        base_path,
        metadata=None,
        enabled=None,
        dpi=None,
        frame_rate=5,
        show=False
        ):
        self.env =env 
        head, _ = os.path.split(base_path)
        self.video_path = base_path+'.mp4' 
        self.frame_dir_path = os.path.join(head,'frames')
        self.save_kwargs = {'frame_rate':frame_rate,'save_path':self.frame_dir_path, 'dpi':dpi}
        self.show = show

    def capture_frame(self,FO_link=None):
        self.env.render(FO_link=FO_link,show=self.show,save_kwargs = self.save_kwargs) 

    def close(self,rm_frames=True):
        fr = str(2*self.save_kwargs['frame_rate'])
        subprocess.call([
            'ffmpeg',
            '-y', # overwrite 
            '-framerate',fr, # how often change to next frame of image
            '-i', os.path.join(self.frame_dir_path,'frame_%04d.png'), # frame images for video geneartion
            '-r', fr, # frame rate of video
            '-pix_fmt', 'yuv420p',
            '-crf','25', # quality of video, lower means better
            '-s', '640x480',
            self.video_path
        ],stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT)   
        if rm_frames:
            subprocess.call(['rm','-rf',self.frame_dir_path])
        self.env.close()




if __name__ == '__main__':
    from zonopy.environments.arm_2d import Arm_2D
    import torch 
    import time 
    env = Arm_2D(n_obs=2)
    video_folder = 'video_test'
    #os.makedirs(video_folder, exist_ok=True)

    

    for i in range(2):
        base_path = os.path.join(video_folder,f'video_{i}')
        video_recorder = VideoRecorder(env,base_path)
        ts = time.time()
        for t in range(10):
            env.step(torch.rand(2))
            video_recorder.capture_frame()
        video_recorder.close()
        env.reset()

    print(f'Time elasped: {time.time()-ts}')
