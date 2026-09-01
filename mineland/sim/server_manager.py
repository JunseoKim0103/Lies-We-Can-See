'''
ServerManager is used to manage the server process.
'''

import subprocess
import shlex
import threading
import time
import os
import shutil

from ..utils import blue_text, red_text

std_print = print
def print(*args, end='\n'):
    text = [blue_text(str(arg)) for arg in args]
    std_print("[Server]", *text, end=end)
def print_error(*args, end='\n'):
    text = [red_text(str(arg)) for arg in args]
    std_print("[Server Error]", *text, end=end)

# TODO: 多线程管理！ServerManager和MineflayerManager都需要多线程管理！
#       特别是 is_running 和 is_runtick_finished 这两个变量，如果数据不一致的话，会导致整个服务器卡死！

class ServerManager:
    def __init__(
        self,
        max_memory="2G",
        wait_interval: float = 0.1,
        is_printing_server_info: bool = False,
        instance_id: int = 0,
        server_port: int = 25565,
    ):
        '''
        Initialize the server manager.

        Args:
            path (str): The path of the server.
            max_memory (str): The maximum memory of the server.
            wait_interval (float): The minimum unit of time for waiting for the server to start and complete a tick.
            is_printing_server_info (bool): Whether to print the server information.
            instance_id (int): Index used to isolate concurrent servers in one container.
                instance_id == 0 uses the shared template server dir (backward compatible);
                instance_id > 0 runs in /tmp/mineland_inst_{id}/server with its own world/port.
            server_port (int): The Minecraft server port written into server.properties.
        '''
        # The shared template dir holds the jar, mods, world templates, etc.
        self.template_dir = os.path.join(os.path.dirname(__file__), 'server')
        self.instance_id = instance_id
        self.server_port = server_port
        if instance_id == 0:
            self.path = self.template_dir
        else:
            self.path = os.path.join('/tmp', f'mineland_inst_{instance_id}', 'server')
            self._setup_instance_dir()
        self.max_memory = max_memory
        if max_memory != "2G":
            assert False, "TODO: Only 2G memory is supported for now"
        self.process = None
        self.thread = None
        self.outputs = []
        self.is_running = False
        self.is_runtick_finished = True
        self.wait_interval = wait_interval
        self.is_printing_server_info = is_printing_server_info
        
        # Game result tracking
        self.game_result = None
        self.game_result_message = None

        self.output_filter = [
            "[Server thread/INFO]: Server unpaused.",
            "[Server thread/INFO]: commands.pause.pausing",
            "[Server thread/INFO]: Server paused.",
            "[Server thread/INFO]: Saving and pausing game...",
            "[Server thread/INFO]: Saving chunks for level",
            # "[Server thread/INFO]: runtick command started",
            "[Server thread/INFO]: runtick command is finished",
            "commands.pause.unpausing",
            "Teleported",
        ]

        self.another_server_is_running_filter = [
            'Perhaps a server is already running on that port'
        ]

    # Files/dirs that each concurrent instance must own privately rather than share.
    # Everything else in the template dir is symlinked (jar, mods, libraries, world templates...).
    _PER_INSTANCE_ENTRIES = {
        'world', 'logs', 'crash-reports', 'session.lock', 'server.properties',
        'ops.json', 'usercache.json', 'whitelist.json',
        'banned-ips.json', 'banned-players.json',
    }

    def _setup_instance_dir(self):
        '''Build a private working dir for instance_id > 0.

        Heavy static content (jar, mods, libraries, world templates) is symlinked
        from the template dir; per-instance state is copied/generated so concurrent
        servers in the same container never touch each other's files.
        '''
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)

        for entry in os.listdir(self.template_dir):
            if entry in self._PER_INSTANCE_ENTRIES:
                continue
            os.symlink(
                os.path.join(self.template_dir, entry),
                os.path.join(self.path, entry),
            )

        # Copy the small mutable state files the server reads at startup.
        for entry in ('ops.json', 'usercache.json', 'whitelist.json',
                      'banned-ips.json', 'banned-players.json'):
            src = os.path.join(self.template_dir, entry)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(self.path, entry))

        self._write_server_properties()

    def _write_server_properties(self):
        '''Generate this instance's server.properties with its own ports.'''
        src = os.path.join(self.template_dir, 'server.properties')
        with open(src, 'r') as f:
            lines = f.readlines()
        rcon_port = self.server_port + 10
        replacements = {
            'server-port': str(self.server_port),
            'query.port': str(self.server_port),
            'rcon.port': str(rcon_port),
        }
        out = []
        for line in lines:
            key = line.split('=', 1)[0].strip()
            if key in replacements:
                out.append(f"{key}={replacements[key]}\n")
            else:
                out.append(line)
        with open(os.path.join(self.path, 'server.properties'), 'w') as f:
            f.writelines(out)

    def select_to_construction_world(self, folder: str = "amongus") :
        # src_folder = os.path.join(self.template_dir, 'embodied_deception') ### embodied_deception
        src_folder = os.path.join(self.template_dir, folder)
        dst_folder = os.path.join(self.path, 'world')
        if not os.path.exists(src_folder):
            return
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.copytree(src_folder, dst_folder)
        pass


    def select_to_normal_world(self) :
        src_folder = os.path.join(self.template_dir, 'normal_world')
        dst_folder = os.path.join(self.path, 'world')
        if not os.path.exists(src_folder):
            return
        if os.path.exists(dst_folder):
            shutil.rmtree(dst_folder)
        shutil.copytree(src_folder, dst_folder)
        pass

    def start(self):
        if not self.process or self.process.poll() is not None:
            command = f"java -Xmx8G -jar fabric-server-mc.1.19-loader.0.14.18-launcher.0.11.2.jar nogui"
            args = shlex.split(command)

            self.process = subprocess.Popen(
                args,
                cwd=self.path,
                shell=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )

            # TODO: Lack of exception handling
            
            self.thread = threading.Thread(target=self.listen_outputs)
            if os.name == 'nt':
                self.thread.setDaemon(True)
            self.thread.start()

    def shutdown(self):
        if self.process and self.process.poll() is None:

            # TODO: Send stop command to server (instead of terminate!)
            self.execute("stop")

            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
                self.process.wait(timeout=5)

    def execute(self, command):
        if self.process and self.process.poll() is None:
            try:
                self.process.stdin.write(f"{command}\n".encode()) 
                self.process.stdin.flush()
                print(f"Command '{command}' is sent successfully.")
                return("ok")
            except Exception as e:
                print_error(f"Command '{command}' is failed. Exception: {str(e)}")
                return "", "Exception: " + str(e)
        else:
            print_error(f"Command '{command}' is failed. Server is not running.")
            return "", "Server not running"
    
    def listen_outputs(self) :
        if self.is_printing_server_info:
            print('Start to listen outputs.')
        
        while True and self.process:
            output = self.process.stdout.readline()
            output_str = output.decode('utf-8', errors='ignore')
            
            # Check if process has ended
            if output == b'' and self.process.poll() is not None:
                break
            
            # should be an independent if statement
            if 'Done' in output_str:
                self.is_running = True
            if 'runtick command is finished now' in output_str:
                self.is_runtick_finished = True
            
            # Track game results (Imposter win, Crewmate win, etc.)
            # Matches both tellraw (raw text) and say command ("[Server] ..." prefix)
            if any(k in output_str for k in ('Imposter Win!', 'Imposters Win!', 'Imposter win', 'Imposters win')):
                self.game_result = 'imposter'
                self.game_result_message = output_str.strip()
            elif any(k in output_str for k in ('Crew Mission Win!', 'Crew mission win')):
                self.game_result = 'crewmate_mission'
                self.game_result_message = output_str.strip()
            elif any(k in output_str for k in ('Crew Win!', 'Crew win', 'Crewmate Win!', 'Crewmates Win!', 'Crewmate win', 'Crewmates win')):
                self.game_result = 'crewmate'
                self.game_result_message = output_str.strip()

            if output == '' or output.isspace() or output_str == '' or output_str.isspace():
                continue
            elif any(substr in output_str for substr in self.output_filter):
                continue
            elif any(substr in output_str for substr in self.another_server_is_running_filter):
                print_error('Another server is running! You may need to restart MineLand.')

            # Format server output nicely
            formatted_output = self._format_server_output(output_str)
            
            if self.is_printing_server_info:
                print(formatted_output, end='')
            self.outputs.append(formatted_output)
    
    def _format_server_output(self, output_str):
        """Format server output for better readability"""
        import re
        
        # Extract tellraw messages and format them specially
        tellraw_pattern = r'\[Server thread/INFO\]: <(.+?)> (.+)'
        tellraw_match = re.match(tellraw_pattern, output_str.strip())
        
        if tellraw_match:
            player_name = tellraw_match.group(1)
            message = tellraw_match.group(2)
            return f"[Server: Chat] {player_name}: {message}\n"
        
        # Format server command execution
        if '[Server thread/INFO]: [' in output_str and 'Executed' in output_str:
            return f"[Server: Console] {output_str.strip()}\n"
        
        # Format regular server messages
        if '[Server thread/INFO]:' in output_str:
            msg = output_str.replace('[Server thread/INFO]:', '').strip()
            return f"[Server: Console] {msg}\n"
        
        # Keep other messages as-is but clean them up
        return output_str if output_str.endswith('\n') else f"{output_str.strip()}\n"
        
        # Read any remaining output after process ends
        if self.process:
            remaining_output = self.process.stdout.read()
            if remaining_output:
                remaining_str = remaining_output.decode('utf-8', errors='ignore')
                for line in remaining_str.split('\n'):
                    if not line or line.isspace():
                        continue
                    
                    # Track game results in remaining output
                    if 'Imposter Win!' in line or 'Imposters Win!' in line or 'Imposter win' in line or 'Imposters win' in line:
                        self.game_result = 'imposter'
                        self.game_result_message = line.strip()
                    elif 'Crew Mission Win!' in line or 'Crew mission win' in line:
                        self.game_result = 'crewmate_mission'
                        self.game_result_message = line.strip()
                    elif 'Crew Win!' in line or 'Crew win' in line or 'Crewmate Win!' in line or 'Crewmates Win!' in line or 'Crewmate win' in line or 'Crewmates win' in line:
                        self.game_result = 'crewmate'
                        self.game_result_message = line.strip()
                    
                    if self.is_printing_server_info:
                        print(line)
                    if not any(substr in line for substr in self.output_filter):
                        self.outputs.append(line)

        if self.is_printing_server_info:
            print("Server is end.")
    
    def get(self, clear=True) :
        ret = self.outputs[:]
        if clear:
            self.outputs = []
        return ret
    
    def wait_for_running(self):
        while not self.is_running:
            time.sleep(self.wait_interval)
        self.is_running = False
    
    def wait_for_runtick_finish(self):
        while not self.is_runtick_finished:
            time.sleep(self.wait_interval)
        self.is_runtick_finished = False
    
    def get_game_result(self, wait_secs: float = 2.0):
        """
        Get the game result.
        Returns a tuple of (result, message) where result is 'imposter', 'crewmate', or None.
        """
        if self.game_result is not None:
            return self.game_result, self.game_result_message
        # Wait a bit for the server to output the result
        import time
        time.sleep(wait_secs)
        return self.game_result, self.game_result_message
    
    def reset_game_result(self):
        """
        Reset the game result tracking.
        """
        self.game_result = None
        self.game_result_message = None

# manager = ServerManager()
# result = manager.start()
# print('ok start')
# print(result)
# print(result.stdout)

# while(True) : 
#     # pass
#     x=input().strip()
#     message = manager.get()
#     print(message)
#     # out = manager.execute(x)
#     # print('output is: ' + out)
