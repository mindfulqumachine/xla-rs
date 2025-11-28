document.addEventListener("DOMContentLoaded", function () {
    var codeBlocks = document.querySelectorAll("pre code.language-mermaid");
    codeBlocks.forEach(function (block) {
        var content = block.textContent;
        var div = document.createElement("div");
        div.className = "mermaid";
        div.textContent = content;
        block.parentNode.replaceWith(div);
    });
    mermaid.initialize({ startOnLoad: true });
});
