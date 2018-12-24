import os
import logging
import tempfile
import subprocess

log = logging.getLogger(__name__)


class PraatUtil(object):
    def __init__(self, praat_path):
        self.path = praat_path
        log.info("Praat location: {}".format(praat_path))
        if not os.path.exists(praat_path):
            raise ValueError(
                "Praat not found at location: {}".format(praat_path))

    def _run(self, path):
        """Runs the praat script at the given path

        Arguments:
            path {str} -- path of the script to run
        """

        with open(os.devnull, 'w') as dev_null:
            subprocess.run([self.path, "--run", path],
                           check=True, stdout=dev_null)

    def execute_script(self, script_content):
        """Executes the given script content

        Arguments:
            script_content {str} -- List of praat commands
        """
        log.debug("Run script: {}".format(script_content))
        with tempfile.NamedTemporaryFile(suffix='.praat') as temp:
            with open(temp.name, "w") as writer:
                writer.writelines(script_content)
            self._run(temp.name)

    def stop_hann_band_filter(self, source_file, target_file, from_freq, to_freq, smoothing_freq):
        """Applies the (stop) Hann filter

        Arguments:
            source_file {str} -- path to source file
            target_file {str} -- location to write filtered file
            from_freq {int} -- From Frequency
            to_freq {int} -- To Frequence
            smoothing_freq {int} -- Smoothing frequency
        """

        script_content = "Read from file... {}\n".format(
            os.path.abspath(source_file))
        script_content += "Filter (stop Hann band)... {} {} {}\n".format(
            from_freq, to_freq, smoothing_freq)
        script_content += "Write to WAV file... {}\n".format(
            os.path.abspath(target_file))
        self.execute_script(script_content)

    def pass_hann_band_filter(self, source_file, target_file, from_freq, to_freq, smoothing_freq):
        """Applies the (pass) Hann filter

        Arguments:
            source_file {str} -- path to source file
            target_file {str} -- location to write filtered file
            from_freq {int} -- From Frequency
            to_freq {int} -- To Frequence
            smoothing_freq {int} -- Smoothing frequency
        """
        script_content = "Read from file... {}\n".format(
            os.path.abspath(source_file))
        script_content += "Filter (pass Hann band)... {} {} {}\n".format(
            from_freq, to_freq, smoothing_freq)
        script_content += "Write to WAV file... {}\n".format(
            os.path.abspath(target_file))
        self.execute_script(script_content)
