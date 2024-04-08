# chatbot



# Installation

Chương trình ví dụ này cho phép bạn sử dụng các mô hình ngôn ngữ LLaMA khác nhau một cách dễ dàng và hiệu quả. 
Có nhiều tùy chọn khác nhau về cách cài đặt gói llama-cpp: 
- CPU 
- CPU + GPU  
- GPU 

## Windows installation
Yêu cầu để cài đặt ,llama-cpp-python
- git
- python
- cmake
Visual Studio Community (đảm bảo bạn cài đặt ứng dụng này với cài đặt sau)
-- Phát triển máy tính để bàn với C ++
-- Phát triển Python
-- Phát triển nhúng Linux với C ++
```
pip install requirements.txt
```
### 1.Clone git repository của llama.cpp
```
git clone --recursive -j8 https://github.com/abetlen/llama-cpp-python.git
```
 #### Đặt môi trường sau biến
 ### Windows:
```
set FORCE_CMAKE=1
set CMAKE_ARGS=-DLLAMA_CUBLAS=OFF
```
Nếu bạn có GPU NVIDIA, hãy đảm bảo được đặt thành ` DLLAMA_CUBLAS ON`
### Compiling and installing
Bây giờ bạn có thể vào thư mục và cài đặt gói `cd llama-cpp-python`
```
cd llama-cpp-python
python -m pip install -e .
```
#### QUAN TRỌNG: 
Nếu bạn đã cài đặt phiên bản chỉ dành cho CPU của gói, bạn cần cài đặt lại từ đầu: hãy xem xét những điều sau lệnh:
```
!python -m pip install -e . --force-reinstall --no-cache-dir
```
### Sử Dụng:
```
streamlit run bottest.py
```
# Tùy chỉnh
```
import streamlit as st
import streamlit_option_menu
from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
```
```
template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer. """

prompt = PromptTemplate.from_template(template)
```
```
# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
```
### CPU
Ví dụ sử dụng mô hình Phi-2 
```
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="D:\Chatlocal\phi2\phi-2.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
```
```
question = """
Question: A rap battle between Stephen Colbert and John Oliver
"""
llm.invoke(question)
```
```

Stephen Colbert:
Yo, John, I heard you've been talkin' smack about me on your show.
Let me tell you somethin', pal, I'm the king of late-night TV
My satire is sharp as a razor, it cuts deeper than a knife
While you're just a british bloke tryin' to be funny with your accent and your wit.
John Oliver:
Oh Stephen, don't be ridiculous, you may have the ratings but I got the real talk.
My show is the one that people actually watch and listen to, not just for the laughs but for the facts.
While you're busy talkin' trash, I'm out here bringing the truth to light.
Stephen Colbert:
Truth? Ha! You think your show is about truth? Please, it's all just a joke to you.
You're just a fancy-pants british guy tryin' to be funny with your news and your jokes.
While I'm the one who's really makin' a difference, with my sat
```
```
llama_print_timings:        load time =   358.60 ms
llama_print_timings:      sample time =   172.55 ms /   256 runs   (    0.67 ms per token,  1483.59 tokens per second)
llama_print_timings: prompt eval time =   613.36 ms /    16 tokens (   38.33 ms per token,    26.09 tokens per second)
llama_print_timings:        eval time = 10151.17 ms /   255 runs   (   39.81 ms per token,    25.12 tokens per second)
llama_print_timings:       total time = 11332.41 ms
```
```
"\nStephen Colbert:\nYo, John, I heard you've been talkin' smack about me on your show.\nLet me tell you somethin', pal, I'm the king of late-night TV\nMy satire is sharp as a razor, it cuts deeper than a knife\nWhile you're just a british bloke tryin' to be funny with your accent and your wit.\nJohn Oliver:\nOh Stephen, don't be ridiculous, you may have the ratings but I got the real talk.\nMy show is the one that people actually watch and listen to, not just for the laughs but for the facts.\nWhile you're busy talkin' trash, I'm out here bringing the truth to light.\nStephen Colbert:\nTruth? Ha! You think your show is about truth? Please, it's all just a joke to you.\nYou're just a fancy-pants british guy tryin' to be funny with your news and your jokes.\nWhile I'm the one who's really makin' a difference, with my sat"
```
### GPU
Nếu cài đặt với phụ trợ BLAS là chính xác, bạn sẽ thấy một chỉ báo trong thuộc tính mô hình.BLAS = 1

Hai trong số các thông số quan trọng nhất để sử dụng với GPU là:

n_gpu_layers - xác định có bao nhiêu lớp của mô hình được giảm tải vào GPU của bạn.
n_batch - có bao nhiêu mã thông báo được xử lý song song.
Đặt các tham số này một cách chính xác sẽ cải thiện đáng kể Tốc độ đánh giá (xem [wrapper code](https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/llms/llamacpp.py) để biết thêm chi tiết hoặc xem ở phía dưới).

```
n_gpu_layers = -1  # Số lớp để đặt trên GPU. Phần còn lại sẽ ở trên CPU. Nếu bạn không biết có bao nhiêu lớp, bạn có thể sử dụng -1 để chuyển tất cả sang GPU.
n_batch = 512  # Nên nằm trong khoảng từ 1 đến n_ctx, hãy xem xét lượng VRAM trong GPU của bạn.

# Đảm bảo đường dẫn mô hình chính xác cho hệ thống của bạn!
llm = LlamaCpp(
    model_path="/Users/rlm/Desktop/Code/llama.cpp/models/openorca-platypus2-13b.gguf.q4_0.bin",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
```
```
llm_chain = prompt | llm
question = "What NFL team won the Super Bowl in the year Justin Bieber was born?"
llm_chain.invoke({"question": question})
```
```
1. Identify Justin Bieber's birth date: Justin Bieber was born on March 1, 1994.

2. Find the Super Bowl winner of that year: The NFL season of 1993 with the Super Bowl being played in January or of 1994.

3. Determine which team won the game: The Dallas Cowboys faced the Buffalo Bills in Super Bowl XXVII on January 31, 1993 (as the year is mis-labelled due to a error). The Dallas Cowboys won this matchup.

So, Justin Bieber was born when the Dallas Cowboys were the reigning NFL Super Bowl.
```
```
llama_print_timings:        load time =   427.63 ms
llama_print_timings:      sample time =   115.85 ms /   164 runs   (    0.71 ms per token,  1415.67 tokens per second)
llama_print_timings: prompt eval time =   427.53 ms /    45 tokens (    9.50 ms per token,   105.26 tokens per second)
llama_print_timings:        eval time =  4526.53 ms /   163 runs   (   27.77 ms per token,    36.01 tokens per second)
llama_print_timings:       total time =  5293.77 ms
```






## Context Management

Trong quá trình tạo văn bản, các mô hình LLaMA có kích thước ngữ cảnh giới hạn, có nghĩa là chúng chỉ có thể xem xét một số lượng mã thông báo nhất định từ văn bản đầu vào và được tạo. Khi ngữ cảnh đầy, mô hình sẽ đặt lại nội bộ, có khả năng mất một số thông tin từ đầu cuộc trò chuyện hoặc hướng dẫn. Các tùy chọn quản lý bối cảnh giúp duy trì tính liên tục và mạch lạc trong những tình huống này.

### Context Size

Tùy chọn `--ctx-size` cho phép bạn đặt kích thước của ngữ cảnh lời nhắc được sử dụng bởi các mô hình LLaMA trong quá trình tạo văn bản. Kích thước ngữ cảnh lớn hơn giúp mô hình hiểu rõ hơn và tạo phản hồi cho dữ liệu đầu vào hoặc cuộc hội thoại dài hơn.

-  `-c N, --ctx-size N`: Đặt kích thước của bối cảnh nhắc nhở (mặc định: 512). Các mô hình LLaMA được xây dựng với bối cảnh năm 2048, sẽ mang lại kết quả tốt nhất khi đầu vào/suy luận dài hơn. Tuy nhiên, việc tăng quy mô bối cảnh vượt quá năm 2048 có thể dẫn đến những kết quả khó lường.
### Extended Context Size

Một số mô hình tinh chỉnh đã mở rộng độ dài ngữ cảnh bằng cách chia tỷ lệ RoPE. Ví dụ: nếu mô hình được đào tạo trước ban đầu có độ dài ngữ cảnh (độ dài chuỗi tối đa) là 4096 (4k) và mô hình tinh chỉnh có 32k. Đó là hệ số tỷ lệ là 8 và sẽ hoạt động bằng cách đặt `--ctx-size` ở trên thành 32768 (32k) và `--rope-scale` thành 8.

-   `--rope-scale N`: Trong đó N là hệ số tỷ lệ tuyến tính được sử dụng bởi mô hình tinh chỉnh.

### Keep Prompt

Tùy chọn `--keep` cho phép người dùng giữ lại lời nhắc ban đầu khi mô hình hết ngữ cảnh, đảm bảo duy trì kết nối với hướng dẫn ban đầu hoặc chủ đề hội thoại.

- `--keep N`: Chỉ định số lượng mã thông báo từ dấu nhắc ban đầu sẽ được giữ lại khi mô hình đặt lại bối cảnh bên trong của nó. Theo mặc định, giá trị này được đặt thành 0 (có nghĩa là không có mã thông báo nào được giữ lại). Sử dụng `-1` để giữ lại tất cả mã thông báo từ dấu nhắc ban đầu.

Bằng cách sử dụng các tùy chọn quản lý ngữ cảnh như `--ctx-size` và `--keep`, bạn có thể duy trì sự tương tác mạch lạc và nhất quán hơn với các mô hình LLaMA, đảm bảo rằng văn bản được tạo vẫn phù hợp với lời nhắc hoặc cuộc hội thoại ban đầu.

## Generation Flags

Các tùy chọn sau cho phép bạn kiểm soát quá trình tạo văn bản và tinh chỉnh tính đa dạng, tính sáng tạo và chất lượng của văn bản được tạo theo nhu cầu của bạn. Bằng cách điều chỉnh các tùy chọn này và thử nghiệm các kết hợp giá trị khác nhau, bạn có thể tìm thấy cài đặt tốt nhất cho trường hợp sử dụng cụ thể của mình.

### Number of Tokens to Predict

- `-n N, --n-predict N`: Đặt số lượng token cần dự đoán khi tạo văn bản (mặc định: 128, -1 = vô cực, -2 = cho đến khi ngữ cảnh được lấp đầy)

Tùy chọn `--n-predict` kiểm soát số lượng mã thông báo mà mô hình tạo ra để phản hồi lời nhắc đầu vào. Bằng cách điều chỉnh giá trị này, bạn có thể tác động đến độ dài của văn bản được tạo. Giá trị cao hơn sẽ tạo ra văn bản dài hơn, trong khi giá trị thấp hơn sẽ tạo ra văn bản ngắn hơn.

Giá trị -1 sẽ cho phép tạo văn bản vô hạn, mặc dù chúng ta có cửa sổ ngữ cảnh hữu hạn. Khi cửa sổ ngữ cảnh đầy, một số mã thông báo trước đó (một nửa số mã thông báo sau `--n-keep`) sẽ bị loại bỏ. Bối cảnh sau đó phải được đánh giá lại trước khi việc tạo có thể tiếp tục. Trên các mô hình lớn và/hoặc cửa sổ ngữ cảnh lớn, điều này sẽ dẫn đến tạm dừng đầu ra đáng kể.

Nếu việc tạm dừng là không mong muốn, giá trị -2 sẽ dừng tạo ngay lập tức khi bối cảnh được lấp đầy.

Điều quan trọng cần lưu ý là văn bản được tạo có thể ngắn hơn số lượng mã thông báo được chỉ định nếu gặp phả iEnd-of-Sequencei (EOS) hoặc lời nhắc ngược lại. Ở chế độ tương tác, quá trình tạo văn bản sẽ tạm dừng và quyền kiểm soát sẽ được trả lại cho người dùng. Ở chế độ không tương tác, chương trình sẽ kết thúc. Trong cả hai trường hợp, việc tạo văn bản có thể dừng trước khi đạt đến giá trị `n-predict` được chỉ định. Nếu bạn muốn mô hình tiếp tục hoạt động mà không bao giờ tự tạo ra Kết thúc chuỗi, bạn có thể sử dụng tham số `--ignore-eos`.

### Temperature

- `--temp N`: Điều chỉnh độ ngẫu nhiên của văn bản được tạo ra (mặc định: 0,8).

Nhiệt độ là một siêu tham số kiểm soát tính ngẫu nhiên của văn bản được tạo ra. Nó ảnh hưởng đến khả năng phân phối xác suất của mã thông báo đầu ra của mô hình. Nhiệt độ cao hơn (ví dụ: 1,5) làm cho đầu ra ngẫu nhiên và sáng tạo hơn, trong khi nhiệt độ thấp hơn (ví dụ: 0,5) làm cho đầu ra tập trung, xác định và thận trọng hơn. Giá trị mặc định là 0,8, mang lại sự cân bằng giữa tính ngẫu nhiên và tính tất định. Ở mức cao nhất, nhiệt độ bằng 0 sẽ luôn chọn mã thông báo tiếp theo có khả năng xảy ra nhất, dẫn đến kết quả đầu ra giống hệt nhau trong mỗi lần chạy.

Cách sử dụng ví dụ: `--temp 0.5`

### Repeat Penalty

- `--repeat-penalty N`: Kiểm soát sự lặp lại của chuỗi mã thông báo trong văn bản được tạo (mặc định: 1.1).
- `--repeat-last-n N`: N token cuối cùng cần xem xét để xử phạt việc lặp lại (mặc định: 64, 0 = bị vô hiệu hóa, -1 = ctx-size).
- `--no-penalize-nl`: Vô hiệu hóa hình phạt đối với mã thông báo dòng mới khi áp dụng hình phạt lặp lại.

Tùy chọn `repeat-penalty` giúp ngăn mô hình tạo ra văn bản lặp đi lặp lại hoặc đơn điệu. Giá trị cao hơn (ví dụ: 1,5) sẽ xử phạt việc lặp lại mạnh mẽ hơn, trong khi giá trị thấp hơn (ví dụ: 0,9) sẽ nhẹ nhàng hơn. Giá trị mặc định là 1.1.

Tùy chọn `repeat-last-n` kiểm soát số lượng token trong lịch sử để xem xét xử phạt việc lặp lại. Giá trị lớn hơn sẽ xem xét xa hơn trong văn bản được tạo để tránh lặp lại, trong khi giá trị nhỏ hơn sẽ chỉ xem xét các mã thông báo gần đây. Giá trị 0 sẽ vô hiệu hóa hình phạt và giá trị -1 đặt số lượng mã thông báo được coi là bằng kích thước ngữ cảnh (`ctx-size`).

Sử dụng tùy chọn `--no-penalize-nl` để vô hiệu hóa hình phạt dòng mới khi áp dụng hình phạt lặp lại. Tùy chọn này đặc biệt hữu ích để tạo các cuộc trò chuyện, đối thoại, mã, thơ hoặc bất kỳ văn bản nào mà mã thông báo dòng mới đóng vai trò quan trọng trong cấu trúc và định dạng. Việc tắt hình phạt dòng mới giúp duy trì luồng tự nhiên và định dạng dự định trong các trường hợp sử dụng cụ thể này.

Ví dụ sử dụng: `--repeat-penalty 1.15 --repeat-last-n 128 --no-penalize-nl`

### Top-K Sampling

- `--top-k N`: Giới hạn lựa chọn mã thông báo tiếp theo ở K mã thông báo có khả năng xảy ra cao nhất (mặc định: 40).

Lấy mẫu Top-k là phương pháp tạo văn bản chỉ chọn mã thông báo tiếp theo từ k mã thông báo có khả năng xảy ra cao nhất được mô hình dự đoán. Nó giúp giảm rủi ro tạo ra các mã thông báo có xác suất thấp hoặc vô nghĩa, nhưng nó cũng có thể hạn chế tính đa dạng của đầu ra. Giá trị cao hơn cho top-k (ví dụ: 100) sẽ xem xét nhiều mã thông báo hơn và dẫn đến văn bản đa dạng hơn, trong khi giá trị thấp hơn (ví dụ: 10) sẽ tập trung vào các mã thông báo có khả năng xảy ra nhất và tạo ra văn bản thận trọng hơn. Giá trị mặc định là 40.

Example usage: `--top-k 30`

### Top-P Sampling

- `--top-p N`: Giới hạn lựa chọn mã thông báo tiếp theo ở một tập hợp con mã thông báo có xác suất tích lũy trên ngưỡng P (mặc định: 0,9).

Lấy mẫu top-p, còn được gọi là lấy mẫu hạt nhân, là một phương pháp tạo văn bản khác chọn mã thông báo tiếp theo từ một tập hợp con các mã thông báo cùng có xác suất tích lũy ít nhất là p. Phương pháp này cung cấp sự cân bằng giữa tính đa dạng và chất lượng bằng cách xem xét cả xác suất của mã thông báo và số lượng mã thông báo để lấy mẫu. Giá trị cao hơn cho top-p (ví dụ: 0,95) sẽ dẫn đến văn bản đa dạng hơn, trong khi giá trị thấp hơn (ví dụ: 0,5) sẽ tạo ra văn bản tập trung và thận trọng hơn. Giá trị mặc định là 0,9.

Example usage:  `--top-p 0.95`

### Lấy mẫu P tối thiểu

- `--min-p N`: Đặt ngưỡng xác suất cơ sở tối thiểu để lựa chọn mã thông báo (mặc định: 0,05).

Phương pháp lấy mẫu Min-P được thiết kế để thay thế cho Top-P và nhằm mục đích đảm bảo sự cân bằng giữa chất lượng và sự đa dạng. Tham số *p* thể hiện xác suất tối thiểu để một mã thông báo được xem xét, liên quan đến xác suất của mã thông báo có nhiều khả năng nhất. Ví dụ: với *p*=0,05 và mã thông báo có nhiều khả năng nhất có xác suất là 0,9, các nhật ký có giá trị nhỏ hơn 0,045 sẽ được lọc ra.

Example usage:  `--min-p 0.05`

### Lấy mẫu miễn phí đuôi (TFS)

- `--tfs N`: Cho phép lấy mẫu không có đuôi với tham số z (mặc định: 1.0, 1.0 = bị tắt).

Lấy mẫu không có đuôi (TFS) là một kỹ thuật tạo văn bản nhằm mục đích giảm tác động của các mã thông báo ít có khả năng xảy ra hơn, có thể ít liên quan hơn, kém mạch lạc hơn hoặc vô nghĩa đối với đầu ra. Tương tự như Top-P, nó cố gắng xác định phần lớn các mã thông báo có khả năng nhất một cách linh hoạt. Nhưng TFS lọc các bản ghi dựa trên đạo hàm thứ hai của xác suất của chúng. Việc thêm mã thông báo sẽ dừng sau khi tổng các đạo hàm thứ hai đạt đến tham số z. Tóm lại: TFS xem xét xác suất của các mã thông báo giảm nhanh như thế nào và cắt bỏ phần đuôi của các mã thông báo không chắc chắn bằng cách sử dụng tham số z. Giá trị điển hình của z nằm trong khoảng từ 0,9 đến 0,95. Giá trị 1.0 sẽ bao gồm tất cả các mã thông báo và do đó vô hiệu hóa tác dụng của TFS.

Example usage:  `--tfs 0.95`

### Lấy mẫu điển hình cục bộ

- `--điển hình N`: Cho phép lấy mẫu điển hình cục bộ với tham số p (mặc định: 1.0, 1.0 = tắt).

Việc lấy mẫu điển hình cục bộ thúc đẩy việc tạo ra văn bản đa dạng và mạch lạc theo ngữ cảnh bằng cách lấy mẫu các mã thông báo điển hình hoặc được mong đợi dựa trên ngữ cảnh xung quanh. Bằng cách đặt tham số p trong khoảng từ 0 đến 1, bạn có thể kiểm soát sự cân bằng giữa việc tạo ra văn bản mạch lạc và đa dạng cục bộ. Giá trị gần hơn 1 sẽ quảng bá các mã thông báo mạch lạc theo ngữ cảnh hơn, trong khi giá trị gần hơn 0 sẽ quảng bá các mã thông báo đa dạng hơn. Giá trị bằng 1 sẽ vô hiệu hóa việc lấy mẫu điển hình cục bộ.

Example usage:  `--điển hình 0,9`

### Lấy mẫu Mirostat

- `--mirostat N`: Bật lấy mẫu Mirostat, kiểm soát sự bối rối trong quá trình tạo văn bản (mặc định: 0, 0 = tắt, 1 = Mirostat, 2 = Mirostat 2.0).
- `--mirostat-lr N`: Đặt tốc độ học của Mirostat, tham số eta (mặc định: 0.1).
- `--mirostat-ent N`: Đặt entropy mục tiêu của Mirostat, tham số tau (mặc định: 5.0).

Mirostat là một thuật toán tích cực duy trì chất lượng văn bản được tạo trong phạm vi mong muốn trong quá trình tạo văn bản. Nó nhằm mục đích đạt được sự cân bằng giữa sự mạch lạc và đa dạng, tránh đầu ra chất lượng thấp do lặp lại quá nhiều (bẫy nhàm chán) hoặc thiếu mạch lạc (bẫy nhầm lẫn).

Tùy chọn `--mirostat-lr` đặt tốc độ học của Mirostat (eta). Tốc độ học ảnh hưởng đến tốc độ phản hồi của thuật toán với phản hồi từ văn bản được tạo. Tốc độ học thấp hơn sẽ dẫn đến điều chỉnh chậm hơn, trong khi tốc độ học cao hơn sẽ làm cho thuật toán phản ứng nhanh hơn. Giá trị mặc định là `0,1`.

Tùy chọn `--mirostat-ent` đặt entropy mục tiêu của Mirostat (tau), đại diện cho giá trị độ phức tạp mong muốn cho văn bản được tạo. Việc điều chỉnh entropy mục tiêu cho phép bạn kiểm soát sự cân bằng giữa tính mạch lạc và tính đa dạng trong văn bản được tạo. Giá trị thấp hơn sẽ mang lại văn bản tập trung và mạch lạc hơn, trong khi giá trị cao hơn sẽ dẫn đến văn bản đa dạng hơn và có khả năng kém mạch lạc hơn. Giá trị mặc định là `5.0`.

Ví dụ sử dụng: `--mirostat 2 --mirostat-lr 0.05 --mirostat-ent 3.0`

### Độ lệch đăng nhập

- `-l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS`: Sửa đổi khả năng mã thông báo xuất hiện khi hoàn thành văn bản được tạo.

Tùy chọn sai lệch logit cho phép bạn điều chỉnh thủ công khả năng các mã thông báo cụ thể xuất hiện trong văn bản được tạo. Bằng cách cung cấp ID mã thông báo và giá trị thiên vị dương hoặc âm, bạn có thể tăng hoặc giảm xác suất tạo mã thông báo đó.

Ví dụ: sử dụng `--logit-bias 15043+1` để tăng khả năng xảy ra mã thông báo 'Xin chào' hoặc `--logit-bias 15043-1` để giảm khả năng xảy ra của mã thông báo. Sử dụng giá trị vô cực âm, `--logit-bias 15043-inf` đảm bảo rằng mã thông báo `Xin chào` không bao giờ được tạo ra.

Một trường hợp sử dụng thực tế hơn có thể là ngăn chặn việc tạo ra `\code{begin}` và `\code{end}` bằng cách đặt `\` mã thông báo (29905) thành vô cực âm với `-l 29905-inf`. (Điều này là do sự phổ biến của mã LaTeX xuất hiện trong suy luận mô hình LLaMA.)

Cách sử dụng ví dụ: `--logit-bias 29905-inf`

### Lấy mẫu P tối thiểu

- `--min-p N`: Đặt ngưỡng xác suất cơ sở tối thiểu để lựa chọn mã thông báo (mặc định: 0,05).

Phương pháp lấy mẫu Min-P được thiết kế để thay thế cho Top-P và nhằm mục đích đảm bảo sự cân bằng giữa chất lượng và sự đa dạng. Tham số *p* thể hiện xác suất tối thiểu để một mã thông báo được xem xét, liên quan đến xác suất của mã thông báo có nhiều khả năng nhất. Ví dụ: với *p*=0,05 và mã thông báo có nhiều khả năng nhất có xác suất là 0,9, các nhật ký có giá trị nhỏ hơn 0,045 sẽ được lọc ra.

Cách sử dụng ví dụ: `--min-p 0.05`

### Lấy mẫu miễn phí đuôi (TFS)

- `--tfs N`: Cho phép lấy mẫu không có đuôi với tham số z (mặc định: 1.0, 1.0 = bị tắt).

Lấy mẫu không có đuôi (TFS) là một kỹ thuật tạo văn bản nhằm mục đích giảm tác động của các mã thông báo ít có khả năng xảy ra hơn, có thể ít liên quan hơn, kém mạch lạc hơn hoặc vô nghĩa đối với đầu ra. Tương tự như Top-P, nó cố gắng xác định phần lớn các mã thông báo có khả năng nhất một cách linh hoạt. Nhưng TFS lọc các bản ghi dựa trên đạo hàm thứ hai của xác suất của chúng. Việc thêm mã thông báo sẽ dừng sau khi tổng các đạo hàm thứ hai đạt đến tham số z. Tóm lại: TFS xem xét xác suất của các mã thông báo giảm nhanh như thế nào và cắt bỏ phần đuôi của các mã thông báo không chắc chắn bằng cách sử dụng tham số z. Giá trị điển hình của z nằm trong khoảng từ 0,9 đến 0,95. Giá trị 1.0 sẽ bao gồm tất cả các mã thông báo và do đó vô hiệu hóa tác dụng của TFS.

Cách sử dụng ví dụ: `--tfs 0.95`

### Lấy mẫu điển hình cục bộ

- `--điển hình N`: Cho phép lấy mẫu điển hình cục bộ với tham số p (mặc định: 1.0, 1.0 = tắt).

Việc lấy mẫu điển hình cục bộ thúc đẩy việc tạo ra văn bản đa dạng và mạch lạc theo ngữ cảnh bằng cách lấy mẫu các mã thông báo điển hình hoặc được mong đợi dựa trên ngữ cảnh xung quanh. Bằng cách đặt tham số p trong khoảng từ 0 đến 1, bạn có thể kiểm soát sự cân bằng giữa việc tạo ra văn bản mạch lạc và đa dạng cục bộ. Giá trị gần hơn 1 sẽ quảng bá các mã thông báo mạch lạc theo ngữ cảnh hơn, trong khi giá trị gần hơn 0 sẽ quảng bá các mã thông báo đa dạng hơn. Giá trị bằng 1 sẽ vô hiệu hóa việc lấy mẫu điển hình cục bộ.

Cách sử dụng ví dụ: `--điển hình 0,9`

### Lấy mẫu Mirostat

- `--mirostat N`: Bật lấy mẫu Mirostat, kiểm soát sự bối rối trong quá trình tạo văn bản (mặc định: 0, 0 = tắt, 1 = Mirostat, 2 = Mirostat 2.0).
- `--mirostat-lr N`: Đặt tốc độ học của Mirostat, tham số eta (mặc định: 0.1).
- `--mirostat-ent N`: Đặt entropy mục tiêu của Mirostat, tham số tau (mặc định: 5.0).

Mirostat là một thuật toán tích cực duy trì chất lượng văn bản được tạo trong phạm vi mong muốn trong quá trình tạo văn bản. Nó nhằm mục đích đạt được sự cân bằng giữa sự mạch lạc và đa dạng, tránh đầu ra chất lượng thấp do lặp lại quá nhiều (bẫy nhàm chán) hoặc thiếu mạch lạc (bẫy nhầm lẫn).

Tùy chọn `--mirostat-lr` đặt tốc độ học của Mirostat (eta). Tốc độ học ảnh hưởng đến tốc độ phản hồi của thuật toán với phản hồi từ văn bản được tạo. Tốc độ học thấp hơn sẽ dẫn đến điều chỉnh chậm hơn, trong khi tốc độ học cao hơn sẽ làm cho thuật toán phản ứng nhanh hơn. Giá trị mặc định là `0,1`.

Tùy chọn `--mirostat-ent` đặt entropy mục tiêu của Mirostat (tau), đại diện cho giá trị độ phức tạp mong muốn cho văn bản được tạo. Việc điều chỉnh entropy mục tiêu cho phép bạn kiểm soát sự cân bằng giữa tính mạch lạc và tính đa dạng trong văn bản được tạo. Giá trị thấp hơn sẽ mang lại văn bản tập trung và mạch lạc hơn, trong khi giá trị cao hơn sẽ dẫn đến văn bản đa dạng hơn và có khả năng kém mạch lạc hơn. Giá trị mặc định là `5.0`.

Ví dụ sử dụng: `--mirostat 2 --mirostat-lr 0.05 --mirostat-ent 3.0`

### Độ lệch đăng nhập

- `-l TOKEN_ID(+/-)BIAS, --logit-bias TOKEN_ID(+/-)BIAS`: Sửa đổi khả năng mã thông báo xuất hiện khi hoàn thành văn bản được tạo.

Tùy chọn sai lệch logit cho phép bạn điều chỉnh thủ công khả năng các mã thông báo cụ thể xuất hiện trong văn bản được tạo. Bằng cách cung cấp ID mã thông báo và giá trị thiên vị dương hoặc âm, bạn có thể tăng hoặc giảm xác suất tạo mã thông báo đó.

Ví dụ: sử dụng `--logit-bias 15043+1` để tăng khả năng xảy ra mã thông báo 'Xin chào' hoặc `--logit-bias 15043-1` để giảm khả năng xảy ra của mã thông báo. Sử dụng giá trị vô cực âm, `--logit-bias 15043-inf` đảm bảo rằng mã thông báo `Xin chào` không bao giờ được tạo ra.

Một trường hợp sử dụng thực tế hơn có thể là ngăn chặn việc tạo ra `\code{begin}` và `\code{end}` bằng cách đặt `\` mã thông báo (29905) thành vô cực âm với `-l 29905-inf`. (Điều này là do sự phổ biến của mã LaTeX xuất hiện trong suy luận mô hình LLaMA.)

Cách sử dụng ví dụ: `--logit-bias 29905-inf`




