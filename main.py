import RAG

if __name__ == "__main__":

    # embedding_model = "BAAI/bge-m3"
    embedding_model = "Alibaba-NLP/gte-Qwen2-1.5B-instruct"

    embedding_path = "ieee/ieee_embedding_1.5B"
    # embedding_path = "ieee/ieee_embedding_1.5B_SemanticDoubleMerge"

    data = "ieee"





    rag = RAG.RAG(data, embedding_model)

    # loaded_documents = rag.load_documents()

    vector_index = rag.embeddingRetrieval(embedding_path)
    rag.chatEngine(vector_index)


    '''
    1.5B embedding model
    Enter your question: what does Percy do while at camp?

    Based on the provided text, here's what Percy Jackson does at Camp Half-Blood:

    * He arrives at the camp after being rescued from the sea (document 1).
    * Chiron greets him and gives him a tour of the camp (document 1).
    * Mr. D (Dionysus) introduces himself as the director of the camp and explains that Percy is a demigod, son of Poseidon (document 2).
    * Percy meets other demigods at the camp, including Grover and Annabeth (document 1).
    * He attends a camp game with Mr. D and Chiron, where he witnesses Mr. D's incredible powers (document 2).

    It appears that Percy is still getting accustomed to his new surroundings and learning about his identity as a demigod while at the camp.


    



    Enter your question: tell me more

    Based on the provided text, here's what else happens at Camp Half-Blood:

    * Percy learns about the existence of gods and demigods from Chiron (document 1).
    * He discovers that his summer camp is actually a training ground for demigods to learn how to control their powers and fight monsters (document 2).
    * Annabeth shows Percy around the camp, introducing him to other demigods and explaining some of the camp's rules (document 1).
    * Luke Castellan, a fellow camper, demonstrates his impressive fighting skills by disassembling dummies with ease (document 2).
    * Mr. D (Dionysus) warns Percy that he must be polite and respectful at the camp, implying that there are strict rules in place (document 2).
    * Chiron explains to Percy that names have power and should not be used carelessly (document 2).

    It seems that Percy is experiencing a whirlwind of new information and experiences as he adjusts to life at Camp Half-Blood. He's learning about 
    his own identity as a demigod, meeting other campers, and getting introduced to the camp's unique culture and rules.
    '''




    # ------------------------------------------------------------------------------------------------





    '''
    m3 SemDoubleMerge

    Enter your question: what does Percy do while at camp?

    According to the provided documents, it seems that Percy is considering staying at Camp Half-Blood year-round and putting 
    in more individual training time with Chiron. He's unsure about what he wants, but it appears that he might be leaning 
    towards staying at the camp.

    In fact, Percy mentions admitting to Annabeth that he'd feel bad about leaving her alone if he were to leave the camp. 
    This suggests that he has a connection with his friends and is thinking about their well-being.

    As for what Percy does while at camp, it's not explicitly stated in the provided documents. However, we can infer that 
    he spends time with Chiron, as they have conversations and Chiron seems to be mentoring him. Percy might also be engaging 
    in activities such as archery, since there is a mention of Chiron galloping away towards the archery range.

    If you'd like more information on what Percy does at camp or his experiences during his time there, I can try to help 
    you find additional context from other sources.







    Enter your question: tell me more

    I'd be happy to provide more information about Percy's activities and experiences at Camp Half-Blood!

    As mentioned earlier, Percy is considering staying at the camp year-round for individual training with Chiron. 
    This suggests that he might be interested in honing his skills as a demigod warrior.

    Percy also mentions being shocked when Luke crumpled his Coke can and threw it into the creek, which shows 
    that he's aware of the importance of not littering at Camp Half-Blood. The nymphs and naiads are known to 
    take care of the camp's environment, and Percy might be learning about the consequences of polluting their home.

    It's also interesting that Percy is hesitant about leaving Annabeth alone at the camp, which indicates that 
    he has developed strong bonds with his friends. He's worried about her being left without him and Clarisse, 
    which suggests that he values their relationships and wants to protect them.

    If you're interested in learning more about Percy's adventures at Camp Half-Blood or his interactions with other 
    demigods, I can try to find additional context from the book series "Percy Jackson & the Olympians" by Rick Riordan. 
    Just let me know!

    '''





    # ------------------------------------------------------------------------------------------------




    '''
    m3

    Enter your question: what does Percy do while at camp?

    Based on the provided text, here's what Percy does while at Camp Half-Blood:

    * He attends a summer camp for demigods (half-human, half-god)
    * He participates in activities with his cabinmates from the Hermes cabin
    * He is surrounded by his friends and acquaintances from various cabins, including Annabeth's friends from Athena, Grover's satyr buddies, and Clarisse's cabinmates
    * He attends a feast prepared in his honor after returning alive from their quest as one of the first heroes to do so
    * He burns his burial shroud made by the Ares cabin (which was an old bedsheet with smiley faces with X'ed-out eyes and the word LOVER painted on it)
    * He sings along with others around the bonfire, enjoying s'mores and socializing with friends

    Please note that this is a summary based on the provided text, which may not cover all aspects of Percy's activities at Camp Half-Blood.








    Enter your question: tell me more
    Based on the provided text, here's what else Percy does while at Camp Half-Blood:

    * He meets with Luke Castellan, a demigod son of Hermes and a former friend, who has become disillusioned with the gods and is planning to betray him.
    * Luke shows Percy a pit scorpion, which is a poisonous creature that can kill him in 60 seconds if he doesn't get away quickly enough.
    * Luke reveals his plans to leave Camp Half-Blood and join Kronos, the Titan King, who wants to overthrow the Olympian gods.
    * Percy tries to reason with Luke, but ultimately decides to flee from the scorpion and escape the situation.

    It's worth noting that this is a pivotal moment in the story, as it marks a turning point in Percy's relationships with his friends and acquaintances at Camp Half-Blood.
    '''




    # ------------------------------------------------------------------------------------------------


    '''
    1.5B SemanticDoubleMerge

    Enter your question: what is one-bit quantization and how does it relate to SNR?


    I'd be happy to help you with that!

    One-bit quantization refers to a type of quantization where each sample of an analog signal is represented by 
    only one bit, which can either be 0 or 1. This means that the signal is effectively binary-coded, and the number
      of possible values is limited to 2^1 = 2.

    In the context of signal processing, one-bit quantization is often used in applications where power consumption, 
    storage space, or communication bandwidth are critical concerns. For example, in some wireless sensor networks 
    or IoT devices, one-bit quantization can help reduce the data rate and power consumption while still maintaining a 
    reasonable level of accuracy.

    Now, let's talk about SNR (Signal-to-Noise Ratio). SNR is a measure of the ratio between the power of the desired 
    signal (the original signal) and the power of the unwanted noise present in the system. It's typically represented 
    as a decibel value (dB) and is calculated using the following formula:

    SNR = 10 \* log10(Ps / Pn)

    where Ps is the power of the desired signal, and Pn is the power of the noise.

    One-bit quantization can impact the SNR in several ways. Since one-bit quantization reduces the number of possible 
    values for each sample, it can lead to a loss of information and a decrease in the overall dynamic range of the signal.
      This, in turn, can result in a reduction in the SNR.

    However, in some cases, one-bit quantization can also be used to improve the SNR in certain applications. For example,
      in analog-to-digital converters (ADCs), one-bit quantization can be used to reduce the noise floor and increase the
        effective resolution of the ADC. This is because the noise power spectral density (PSD) is proportional to the 
        square of the number of bits, so a reduction in the number of bits can lead to a lower noise floor.

    In summary, one-bit quantization is a type of quantization that represents each sample as either 0 or 1, which 
    can be useful in certain applications where power consumption and storage space are critical. The impact of one-bit 
    quantization on SNR depends on the specific application and system design, but it can potentially reduce the SNR 
    due to information loss or improve it by reducing noise.

    I hope that helps! Do you have any more questions about one-bit quantization or SNR?
        

    





    1.5B

    Enter your question: what is one-bit quantization and how does it relate to SNR?

    Based on the provided documents, here's what I found:

    One-bit quantization refers to a technique where an analog-to-digital converter (ADC) uses only one bit to 
    represent the signal amplitude at a certain time. This means that instead of using multiple bits to capture 
    the full range of signal amplitudes, a one-bit ADC uses a single bit to determine whether the signal is above 
    or below a certain threshold.

    Regarding SNR (Signal-to-Noise Ratio), here's what I found:

    * In [12], it is mentioned that when the signal-to-noise ratio (SNR) is low, the matched filter is still effective 
    in processing one-bit quantized SAR data.

    * In [15], it is stated that the image does not exhibit any appreciable degradation even with one-bit quantization,
      which suggests that SNR plays a role in maintaining image quality.

    It appears that one-bit quantization can be used effectively in radar systems, including SAR and DOA estimation, as 
    long as the SNR is sufficient. When the SNR is low, more sophisticated techniques may be needed to maintain accuracy. 
    However, more research would be required to fully understand the relationship between one-bit quantization and SNR.

    If you'd like me to clarify anything or provide more information, please let me know!

    '''