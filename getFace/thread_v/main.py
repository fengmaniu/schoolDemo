#! python 3
# -*- coding: utf-8 -*-

# This code is heavily based on gist: https://gist.github.com/Nikola-K/8b5b510a5c85c3e207fb
#
# This example of using pyqtSignal and QThread has been updated to work with
# python 3, PyQt5 and urllib.
#
# Modified by evereux@gmail.com
# Created 02-Sep-2016

import json
import sys
import time
import urllib.request

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication

from design import Ui_MainWindow

class getPostsThread(QThread):
    # create the signals that will be emitted from this QThread
    message_to_listsubmissions = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, subreddits):
        """
         Make a new thread instance with the specificed
         subreddits as the first argument. The subreddits argument
         will be stored in an instance list called subreddits
         which then can be accessed by all other class instance functions.
         :param subreddits: A list of subreddit names
         :type subreddits: list
         """
        QThread.__init__(self)
        self.subreddits = subreddits

    def __del__(self):
        self.wait()

    def get_top_post(self, subreddit):
        """
        Return a pre-formatted string with the top post title, author,
        and subreddit name from the subreddit passed as the only required
        argument.
        :param subreddit: A valid subreddit name
        :type subreddit: str
        :return: A string with top post title, author,
                and subreddit name from that subreddit
        :rtype: str
        """
        url = "https://www.reddit.com/r/{}.json?limit=1".format(subreddit)
        headers = {'User-Agent': 'nikolak@outlook.com tutorial code'}
        request = urllib.request.Request(url, headers=headers)
        response = urllib.request.urlopen(request)
        data = json.loads(response.read().decode('utf-8'))
        top_post = data['data']['children'][0]['data']

        return "'{title}' by {author} in {subreddit}".format(**top_post)

    def run(self):
        """ Go over every item in the self.subreddits list
        (which was supplied during __int__
        and for every item assume it's a string with a valid subreddit
        name and fetch the top post using the get_top_post method
        from reddit. Store the result in a local variable named top_post
        and then emit a SIGNAL add_post(QString) where QString is equal
        to the top_post variable that was set by the get_top_post function."""
        for subreddit in self.subreddits:
            top_post = self.get_top_post(subreddit)
            self.message_to_listsubmissions.emit(top_post)
            time.sleep(2)

        self.finished.emit()


class ThreadingTutorial(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(ThreadingTutorial, self).__init__(parent)
        self.setupUi(self)
        self.btn_start.clicked.connect(self.start_getting_top_posts)
        self.show()

    def start_getting_top_posts(self):
        # Get the subreddits user entered into a QLine Edit field
        # this will be equal to '' if there is no text entered
        subreddit_list = self.edit_subreddits.text().split(',')

        if subreddit_list == ['']:  # since ''.split(',') == [''] we use that to check

            # whether there is anything there to fetch from
            # and if not show a message and abort
            QMessageBox.critical(self, "No subreddits", "You didn't enter any subreddits", QMessageBox.Ok)
            return

        # Se the maximum value of progress bar, can be any int and it will
        # be automatically converted to x/100% values
        # e.g. max_value = 3, current_value = 1, the progress bar will show 33%
        self.progressBar.setMaximum(len(subreddit_list))

        # setting the value on every run to 0
        self.progressBar.setValue(0)

        # we have a list of subreddits which we use to create new getPostsThread
        # instance and we pass that to the thread
        self.get_thread = getPostsThread(subreddit_list)

        # Next we need to connect the events from that thread to functions we want
        # to be run when those signals get fired

        # Adding post will be handled in the add_post method and the signal that
        # the thread will emit is message_to_listsubmissions(top_post)
        self.get_thread.message_to_listsubmissions.connect(self.add_post)

        # This is pretty self explanatory
        # regardless of whether the thread finishes or the user terminates it
        # we want to show the notification to the use that adding is done
        # and regardless of whether it was terminated or finished by itself
        # the finished signal will go off. So we don't need to catch the terminated one
        # specically, but we could if we wanted.
        self.get_thread.finished.connect(self.done)

        # we have all the events we need connected we can start the thread
        self.get_thread.start()

        # At this point we want to allow user to stop/terminate the thread
        # so we enable that button..
        self.btn_stop.setEnabled(True)

        # And we connect the click of that button to the built in
        # terminated method that all QThread instances have
        self.btn_stop.clicked.connect(self.get_thread.terminate)

        # we don't want to enable user to start another thread while this one is running
        # so we disable the start button.
        self.btn_start.setEnabled(True)

    def add_post(self, post_text):
        """
        Add the text that's given to this function to the
        list_submissions QListWidget we have in our GUI and
        increase the current value of progress bar by 1
        :param post_text: text of the item to add to the list
        :type post_text: str
        """
        self.list_submissions.addItem(post_text)
        self.progressBar.setValue(self.progressBar.value() + 1)

    def done(self):
        """
        Show the message that fetching posts is done.
        Disable Stop button enable the Start one and reset progress bar to 0.
        :return:
        """
        self.btn_stop.setEnabled(False)
        self.btn_start.setEnabled(True)
        self.progressBar.setValue(0)
        QMessageBox.information(self, "Done!", "Done fetching posts!")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = ThreadingTutorial()
    sys.exit(app.exec_())