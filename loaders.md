# Wiki
## example
inputArray = ['Berlin', 'Rome', 'Tokyo', 'Canberra', 'Santiago']

WikipediaReader = download_loader("WikipediaReader")
loader = WikipediaReader()
year_docs = loader.load_data(pages=[i])


# Youtube
## example
inputArray = ["https://www.youtube.com/watch?v=tkH2-_jMCSk",
    "https://www.youtube.com/watch?v=wTBSGgbIvsY" , # meditation
    "https://www.youtube.com/watch?v=yaWVflQolmM" , # fasting 2
]

YoutubeTranscriptReader = download_loader("YoutubeTranscriptReader")
loader = YoutubeTranscriptReader()
x_docs = loader.load_data(ytlinks=[inputdata])

## Notes
- Document needs to be iterated twice


# PDF
## example
PDFReader = download_loader("PDFReader")
loader = PDFReader()
documents = loader.load_data(file=Path('./llama.pdf'))