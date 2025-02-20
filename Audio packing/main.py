import os
import shutil
import heapq
from abc import ABC, abstractmethod
from pathlib import Path
import time



class Audio:
    def __init__(self, file_name, file_duration):
        self.file_name = file_name
        self.file_duration = file_duration

    def str(self):
        return f"{self.file_name} ({self.file_duration} seconds)"


class Folder:
    def __init__(self, capacity):
        self.capacity = capacity
        self.files = []
        self.total_time = 0

    def remaining_time(self):
        return self.capacity - self.total_time

    def add_audio(self, audio):
        self.files.append(audio)
        self.total_time += audio.file_duration


class Audio_manager:
    def __init__(self, metadata, input_folder):
        self.metadata = metadata
        self.input_folder = input_folder
        self.audiolist = []

    def parse_metadata(self, metadata):
        with open(metadata, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    file_name = parts[0]
                    duration = self.time_to_seconds(parts[1])
                    self.audiolist.append(Audio(file_name, duration))

    @staticmethod
    def time_to_seconds(time_str):
        h, m, s = map(int, time_str.split(':'))
        return h * 3600 + m * 60 + s

    @staticmethod
    def create_output_folders(folders, algorithm_name, output_folder, input_folder):

        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        for folder_index, folder in enumerate(folders):
            folder_name = f"{algorithm_name}Folder{folder_index + 1}"
            folder_path = output_path / folder_name
            folder_path.mkdir(parents=True, exist_ok=True)

            total_duration = folder.total_time

            for audio in folder.files:
                source_file = Path(input_folder) / audio.file_name
                destination_file = folder_path / audio.file_name

                if source_file.is_file():
                    shutil.copy(str(source_file), str(destination_file))  # Convert to str if needed
                    print(f"Copied {audio.file_name} to {folder_name}")
                else:
                    print(f"Error: {audio.file_name} does not exist or destination is invalid.")

            metadata_file_path = folder_path / "metadata.txt"
            with metadata_file_path.open("w") as f:
                f.write(f"Total Duration: {total_duration} seconds\n")
                f.write("Files:\n")
                for audio in folder.files:
                    f.write(f"{audio.file_name}: {audio.file_duration} seconds\n")

        print(f"\nFiles sorted using {algorithm_name} algorithm. Folders and metadata files created.")


class Algorithm(ABC):
    def __init__(self, folder_capacity):
        self.folder_capacity = folder_capacity

    @abstractmethod
    def algo_implmnt(self, audio_files):
        pass


class WorstFitLinear(Algorithm):  # O(1)
    def __init__(self, folder_capacity):  # O(1)
        super().__init__(folder_capacity)  # O(1)

    def algo_implmnt(self, audio_files):  # O(1)+O(N)*O(M)=O(N)*O(M)
        folders = []  # O(1)
        for audio in audio_files:  # O(N)*[O(1)+O(M)+O(1)+O(1)]=O(N)*O(M)
            index = -1  # O(1)
            max_rem = -1  # O(1)

            for i, folder in enumerate(folders):  # O(M)*[O(1)+O(1)]= O(M)
                rem = folder.remaining_time()  # O(1)
                if audio.file_duration <= rem and rem > max_rem:  # O(1)
                    index = i  # O(1)
                    max_rem = rem  # O(1)

            if index == -1:  # O(1)
                new_folder = Folder(self.folder_capacity)  # O(1)
                new_folder.add_audio(audio)  # O(1)
                folders.append(new_folder)  # O(1)
            else:  # O(1)
                folders[index].add_audio(audio)  # O(1)

        return folders  # O(1)


class WorstFitPriorityQueue(Algorithm):
    def __init__(self, folder_capacity):
        super().__init__(folder_capacity)
        self.folder_heap = []

    def algo_implmnt(self, audio_files):  # O(1)+O(N)*O(LOG(M))=O(N)*O(LOG(M))
        folders = []  # O(1)
        for audio in audio_files:  # O(N)*(O(LOG(M))+O(LOG(M))= O(N)*O(LOG(M))
            if self.folder_heap and -self.folder_heap[0][0] >= audio.file_duration:  # O(LOG(M)+O(1)= 0(LOG(M))
                remaining_time, index = heapq.heappop(self.folder_heap)  # O(1)
                folders[index].add_audio(audio)  # O(1)
                heapq.heappush(self.folder_heap, (-folders[index].remaining_time(), index))  # O(LOG(M))
            else:  # O(1)+O(LOG(M))=O(LOG(M))
                new_folder = Folder(self.folder_capacity)  # O(1)
                new_folder.add_audio(audio)  # O(1)
                folders.append(new_folder)  # O(1)
                heapq.heappush(self.folder_heap, (-new_folder.remaining_time(), len(folders) - 1))  # O(LOG(M))
        return folders  # O(1)


class WorstFitDecreasingLinear(Algorithm):
    def __init__(self, folder_capacity):
        super().__init__(folder_capacity)

    def algo_implmnt(self, audio_files):  # O(1)+O(N LOG(N))+[O(N)*O(M)]=O(NLOG(N))+O(N)*O(M)
        folders = []  # O(1)
        audio_files.sort(key=lambda x: x.file_duration, reverse=True)  # O(N LOG(N))
        for audio in audio_files:  # O(N)*[O(1)+O(M)+O(1)+O(1)]= O(N)*O(M)
            index = -1  # O(1)
            max_rem = -1  # O(1)

            for i, folder in enumerate(folders):  # O(M)*[0(1)+O(1)]= O(M)
                rem = folder.remaining_time()  # O(1)
                if audio.file_duration <= rem and rem > max_rem:  # O(1)
                    index = i  # O(1)
                    max_rem = rem  # O(1)

            if index == -1:  # O(1)
                new_folder = Folder(self.folder_capacity)  # O(1)
                new_folder.add_audio(audio)  # O(1)
                folders.append(new_folder)  # O(1)
            else:  # O(1)
                folders[index].add_audio(audio)  # O(1)

        return folders



class WorstFitDecreasingPriorityQueue(Algorithm):
    def __init__(self, folder_capacity):
        super().__init__(folder_capacity)
        self.folder_heap = []

    def algo_implmnt(self, audio_files):
        # O(1)+O(N LOG(N))+[O(N)(LOG(M))]+O(1)= O(N LOG(N))+[O(N)(LOG(M))]
        # total complexity = 0(N LOG(N))+0(N LOG(M))
        # but since we sure that M can never exceed N (the number of folders won't exceed the number of audio files M<=N )
        # Final total complexity= O(N LOG(N)
        folders = []  # O(1)
        audio_files.sort(key=lambda x: x.file_duration, reverse=True)  # O(N LOG(N))
        for audio in audio_files:  # O(N)*O(LOG(M))
            if self.folder_heap and -self.folder_heap[0][0] >= audio.file_duration:  # 0(LOG(M)+0(1)+0(LOG(M)=O(LOG(M))
                remaining_time, index = heapq.heappop(self.folder_heap)  # O(LOG(M))
                folders[index].add_audio(audio)  # O(1)
                heapq.heappush(self.folder_heap, (-folders[index].remaining_time(), index))  # O(LOG(M))
            else:  # 0(LOG(M))
                new_folder = Folder(self.folder_capacity)  # 0(1)
                new_folder.add_audio(audio)  # 0(1)
                folders.append(new_folder)  # 0(1)
                heapq.heappush(self.folder_heap, (-new_folder.remaining_time(), len(folders) - 1))  # 0(LOG(M))
        return folders


class first_fit_decreasing_with_linear_search(Algorithm):  # O(1)+ O(max(NlogN , N*M)) = O(max(NlogN , N*M))
    def __init__(self, folder_capacity):
        super().__init__(folder_capacity)  # O(1)

    def algo_implmnt(self, audio_files):  # O(max(NlogN , N*M))
        sorted_files = sorted(audio_files, key=lambda x: x.file_duration, reverse=True)  # NLogN
        folder_contents = []  # O(1)

        for audio in sorted_files:  # iterations(N)*body --> O(N * M)
            file_name = audio.file_name  # O(1)
            file_duration = audio.file_duration  # O(1)
            placed = False  # O(1)
            if file_duration > self.folder_capacity:  # O(1)
                raise ValueError(f"File '{file_name}' exceeds folder capacity and cannot be placed.")  # O(1)

            for folder in folder_contents:  # iterations(M)*body --> O(M) * O(1) = O(M)
                if folder.total_time + file_duration <= self.folder_capacity:  # O(1)
                    folder.add_audio(audio)  # O(1)
                    placed = True  # O(1)
                    break  # O(1)

            if not placed:  # O(1)
                newfolder = Folder(self.folder_capacity)  # O(1)
                newfolder.add_audio(audio)  # O(1)
                folder_contents.append(newfolder)  # O(1)

        return folder_contents  # O(1)


class best_fit_decreasing_with_priority_queue(Algorithm):
    def __init__(self, folder_capacity):
        super().__init__(folder_capacity)  # O(1)

    def algo_implmnt(self, audio_files):  # O(1)+O(1)+O(N)(O(MlogM) = O(N)(O(MlogM)
        sorted_files = sorted(audio_files, key=lambda x: x.file_duration, reverse=True)  # O(NlogN)

        pq = []  # O(1)
        folder_contents = []  # O(1)

        for audio in sorted_files:  # iterations(N)body --> O(N)(O(MlogM)
            file_name = audio.file_name  # O(1)
            file_duration = audio.file_duration  # O(1)
            if file_duration > self.folder_capacity:  # O(1)
                raise ValueError(f"File '{file_name}' exceeds folder capacity and cannot be placed.")  # O(1)

            if pq:  # O(MlogM + LogM +MlogM)= O(MlogM) "we choose the first body not else (max complexity)"
                temp_pq = []  # O(1)
                best_folder = None  # O(1)
                while pq:  # iterations(M)*body --> O(M)*O(LogM)= O(MlogM)
                    remaining_space, folder_index = heapq.heappop(pq)  # O(logM)
                    if remaining_space >= file_duration:  # O(1)
                        best_folder = (remaining_space, folder_index)  # O(1)
                        break  # O(1)
                    else:
                        temp_pq.append((remaining_space, folder_index))  # O(1)

                for item in temp_pq:  # iterations(M)*body --> O(M)*O(LogM)= O(MlogM)
                    heapq.heappush(pq, item)  # O(logM)

                if best_folder:  # O(1)+O(1)+O(logM) = O(logM) "both are the same"
                    remaining_space, folder_index = best_folder  # O(1)
                    folder_contents[folder_index].add_audio(audio)  # O(1)

                    heapq.heappush(pq, (remaining_space - file_duration, folder_index))  # O(logM)
                else:  # O(1)+O(1)+O(1)+O(logM) = O(logM)
                    newfolder = Folder(self.folder_capacity)  # O(1)
                    newfolder.add_audio(audio)  # O(1)
                    folder_contents.append(newfolder)  # O(1)

                    heapq.heappush(pq, (newfolder.remaining_time(), len(folder_contents) - 1))  # O(logM)
            else:  # O(1)+O(1)+O(1)+O(logM) = O(logM)
                newfolder = Folder(self.folder_capacity)  # O(1)
                newfolder.add_audio(audio)  # O(1)
                folder_contents.append(newfolder)  # O(1)

                heapq.heappush(pq, (newfolder.remaining_time(), len(folder_contents) - 1))  # O(logM)

        return folder_contents  # O(1)


class folder_filling(Algorithm):
    def __init__(self, folder_capacity):
        super().__init__(folder_capacity)

    def algo_implmnt(self,
                     audio_files):  # O(N*M*C) where M(the number of folders) could possibly reach N in the worst case scenario so the algorithm could reach O(N^2*C)
        folders = []  # O(1)
        while audio_files:  # O(M) could reach O(N)
            best_subset, audio_files = self.find_best_subset(audio_files)  # (N*C)
            if not best_subset:  # O(1)
                break  # O(1)
            folder = Folder(self.folder_capacity)  # O(1)
            folder.files = best_subset  # O(1)
            folder.total_time = sum(audio.file_duration for audio in best_subset)  # O(n) in the worst case
            folders.append(folder)  # O(1)
        return folders

    def find_best_subset(self, audio_files):
        n = len(audio_files)  # O(1)
        dp = [0] * (self.folder_capacity + 1)  # O(C)
        file_count = [0] * (self.folder_capacity + 1)  # O(C)
        subset = [[[] for capacity in range(self.folder_capacity + 1)] for list_index in
                  range(2)]  # O(2C) : creates 2 lists inside each list another list of c

        for i in range(n):  # O(N)
            for c in range(self.folder_capacity, audio_files[i].file_duration - 1, -1):  # O(C)
                if (
                        dp[c - audio_files[i].file_duration] + audio_files[i].file_duration > dp[c]
                        or (  # O(1)
                        dp[c - audio_files[i].file_duration] + audio_files[i].file_duration == dp[c]
                        and file_count[c - audio_files[i].file_duration] + 1 > file_count[c]
                )
                ):
                    dp[c] = dp[c - audio_files[i].file_duration] + audio_files[i].file_duration  # O(1)
                    file_count[c] = file_count[c - audio_files[i].file_duration] + 1  # O(1)
                    subset[1][c] = subset[0][c - audio_files[i].file_duration] + [audio_files[i]]  # O(1)

            subset[0] = [row[:] for row in subset[1]]  # (C)

        best_sum = dp[self.folder_capacity]  # for tracing     #(1)

        best_subset = subset[1][self.folder_capacity]  # (1)
        remaining_files = [file for file in audio_files if file not in best_subset]  # (N)
        print(
            f"Optimal packing sum for the folder is {best_sum} seconds out of {self.folder_capacity} seconds.")  # O(1)

        return best_subset, remaining_files


def main():
    metadata_file = r"C:\Users\Sama\Downloads\Complete2\Complete2\AudiosInfo.txt"
    input_folder = r"C:\Users\Sama\Downloads\Complete2\Complete2\Audios"
    output_folder = r"C:\Users\Sama\Downloads\audioouput"

    folder_capacity = int(input("Enter your desired folder capacity (in seconds): "))

    audio_manager = Audio_manager(metadata_file, input_folder)
    audio_manager.parse_metadata(metadata_file)

    print("Choose an algorithm:")
    print("1. First Fit Decreasing with Linear Search")
    print("2. Best Fit Decreasing using Priority Queue")
    print("3. Worst-fit algorithm using Linear Search")
    print("4. Worst-fit algorithm using Priority queue")
    print("5. Worst-fit decreasing algorithm using Linear Search")
    print("6. Worst-fit decreasing algorithm using Priority queue")
    print("7. Folder filling")

    choice = int(input("Enter your choice : "))

    if choice == 1:
        algorithm = first_fit_decreasing_with_linear_search(folder_capacity)
        algorithm_name = "FirstFitDecreasing"
    elif choice == 2:
        algorithm = best_fit_decreasing_with_priority_queue(folder_capacity)
        algorithm_name = "BestFitDecreasing"
    elif choice == 3:
        algorithm = WorstFitLinear(folder_capacity)
        algorithm_name = "WorstFitLinear"
    elif choice == 4:
        algorithm = WorstFitPriorityQueue(folder_capacity)
        algorithm_name = "WorstFitPriorityQueue"
    elif choice == 5:
        algorithm = WorstFitDecreasingLinear(folder_capacity)
        algorithm_name = "WorstFitDecreasingLinear"
    elif choice == 6:
        algorithm = WorstFitDecreasingPriorityQueue(folder_capacity)
        algorithm_name = "WorstFitDecreasingPriorityQueue"
    elif choice == 7:
        algorithm = folder_filling(folder_capacity)
        algorithm_name = "folderfilling"


    else:
        print("Invalid choice.")
        return

    try:
        start = time.time()
        folders = algorithm.algo_implmnt(audio_manager.audiolist)
        audio_manager.create_output_folders(folders, algorithm_name, output_folder, input_folder)
        end = time.time()

    except ValueError as e:
        print(f"Error: {e}")

    print(end-start)


if __name__ == "__main__":
    main()