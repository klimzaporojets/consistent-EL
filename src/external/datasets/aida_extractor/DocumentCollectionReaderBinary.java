package spirit.toolchain2.file.store;

import java.io.BufferedInputStream;
import java.io.DataInput;
import java.io.DataInputStream;
import java.io.EOFException;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;

import spirit.toolchain2.file.store.SimpleDocument.FieldByteArray;

public class DocumentCollectionReaderBinary extends AbstractCollectionReader {
	private FileInputStream fileInputStream;
	private InputStream inputStream;
	private DataInput input;
	private String filename;
	
	public boolean debug = false;

	public DocumentCollectionReaderBinary(InputStream inputStream) throws IOException {
		this.inputStream = inputStream;
		input = new DataInputStream(new BufferedInputStream(inputStream));
	}
	
	public DocumentCollectionReaderBinary(String filename) throws IOException {
		this.filename = filename;
		fileInputStream = new FileInputStream(filename);
		input = new DataInputStream(new BufferedInputStream(fileInputStream));
	}
	
	public DocumentCollectionReaderBinary(DataInput input) {
		this.input = input;
	}
	
	public boolean read(SimpleDocument document) throws IOException {
		int version;
		try { version = input.readInt(); } catch (EOFException e) { return false; }
		if (version == -1) {
			int fields = input.readInt();
			for (int i = 0; i < fields; i++) {
				String fieldName = input.readUTF();
				String fieldValue = input.readUTF();
				document.add(fieldName, fieldValue);
			}
		} else if (version == -2) {
			int fields = input.readInt();
			for (int i = 0; i < fields; i++) {
				String fieldName = input.readUTF();
				int size = input.readInt();
				byte[] data = new byte[size];
				input.readFully(data);
				String fieldValue = new String(data, "UTF-8");
				document.add(fieldName, fieldValue);
			}
		} else if (version == -3) {
			int fields = input.readInt();
			for (int i = 0; i < fields; i++) {
				String fieldName = input.readUTF();
				byte type = input.readByte();
				if (type == 0) {						// string
					int size = input.readInt();
					byte[] data = new byte[size];
					input.readFully(data);
					String fieldValue = new String(data, "UTF-8");
					document.add(fieldName, fieldValue);
				} else if (type == 1) {
					int size = input.readInt();
					byte[] data = new byte[size];
					input.readFully(data);
					
					document.add(new FieldByteArray(fieldName, data));
				}
			}
		} else if (version == -4) {
			int fields = input.readInt();
			for (int i = 0; i < fields; i++) {
				String fieldName = readShortUTF();
				byte type = input.readByte();
				if (type == 0) {						// string
					int size = input.readInt();
					byte[] data = new byte[size];
					input.readFully(data);
					String fieldValue = new String(data, "UTF-8");
					document.add(fieldName, fieldValue);
				} else if (type == 1) {
					int size = input.readInt();
					byte[] data = new byte[size];
					input.readFully(data);
					
					document.add(new FieldByteArray(fieldName, data));
				}
			}
		} else {
			throw new IOException("invalid version: " + version);
		}
//		if (filename != null) {
//			document.add(DocumentCollectionTools.META_FIELD_FILENAME, filename);
//			int sep = filename.lastIndexOf('/');
//			document.add(DocumentCollectionTools.META_FIELD_BASENAME, filename.substring(sep+1));
//		}
		return true;
	}
	
	public String readShortUTF() throws IOException {
		int size = input.readUnsignedShort();
		byte[] data = new byte[size];
		input.readFully(data);
		return new String(data, "UTF-8");
	}

	public void close() throws IOException {
		if (fileInputStream != null) {
			fileInputStream.close();
			fileInputStream = null;
		}
	}
	
//	private void seek(long position) throws IOException {
//		fileInputStream.getChannel().position(position);
//		input = new DataInputStream(new BufferedInputStream(fileInputStream));
//	}
	
	

}
